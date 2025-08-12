#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cublasLt.h>

inline const char* cublasStatusToString(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:           return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:   return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:      return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:     return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:     return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:     return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:    return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:     return "NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:     return "LICENSE_ERROR";
        default:                              return "UNKNOWN_STATUS";
    }
}

#define CUBLASLT_CHECK(expr)                                                     \
  do {                                                                           \
    cublasStatus_t _st = (expr);                                                 \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                          \
      TORCH_CHECK(false,                                                         \
        "cuBLASLt call failed: ", cublasStatusToString(_st),                     \
        " (", static_cast<int>(_st), ") at ", __FILE__, ":", __LINE__);          \
    }                                                                            \
  } while (0)
  
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")


namespace xops {

inline cudaDataType to_cuda_dtype(at::ScalarType t) {
    if (t == at::kFloat)     return CUDA_R_32F;
    if (t == at::kBFloat16)  return CUDA_R_16BF;
    TORCH_CHECK(false, "Only torch.float32 and torch.bfloat16 are supported");
}

inline cublasComputeType_t to_compute_type(at::ScalarType t) {
    if (t == at::kFloat)     return CUBLAS_COMPUTE_32F;
    // BF16 inputs, use fast 16bf (Tensor Cores) with FP32 accumulate
    if (t == at::kBFloat16)  return CUBLAS_COMPUTE_32F_FAST_16BF;
    TORCH_CHECK(false, "Only torch.float32 and torch.bfloat16 are supported");
}

at::Tensor cublaslt_mm_impl(const at::Tensor& A, const at::Tensor& B, const c10::optional<at::Tensor>& bias) {

    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");

    const auto dtype = A.scalar_type();
    TORCH_CHECK((dtype == at::kFloat || dtype == at::kBFloat16), "Only supports float32 or bfloat16; but found ", dtype);
    TORCH_CHECK(A.scalar_type() == dtype && B.scalar_type() == dtype, "A and B must be of the same dtype");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D (matrix)");
    const auto m = A.size(0);
    const auto k = A.size(1);
    TORCH_CHECK(B.size(0) == k, "matmul inner dims must match: A(:,", k, ") vs B(", B.size(0), ",:)");
    const auto n = B.size(1);

    at::Tensor A_ = A.contiguous();
    at::Tensor B_ = B.contiguous();
    at::Tensor bias_;
    
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA if provided");
        TORCH_CHECK(bias->device() == A.device(), "bias must be on same device as A and B");
        TORCH_CHECK(bias->scalar_type() == dtype, "bias must match dtype of A and B");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n,
                    "bias is expected to be 1D of length ", n, ", but got shape ", bias->sizes());
        bias_ = bias->contiguous();  
    }

    auto D = at::empty({m, n}, A_.options()); // output

    // Device guard & stream
    c10::cuda::CUDAGuard guard(A_.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use PyTorch's pooled cuBLASLt handle (don't create/destroy yourself)
    cublasLtHandle_t lt = at::cuda::getCurrentCUDABlasLtHandle();

    const cudaDataType dataType = to_cuda_dtype(dtype);
    const cublasComputeType_t computeType = to_compute_type(dtype);
    const cudaDataType scaleType = CUDA_R_32F; // alpha/beta in float

    // Matmul descriptor, CUBLAS_OP_N means no transpose
    cublasLtMatmulDesc_t opDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
    {
        cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    }

    if (bias) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

        void* bias_ptr = bias_.data_ptr();  // make a variable, pass its address
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }

    // Layouts (row-major)
    cublasLtMatrixLayout_t aLayout, bLayout, cLayout, dLayout;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, dataType, m, k, k)); // ldA
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bLayout, dataType, k, n, n)); // ldB
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, dataType, m, n, n)); // ldC // input C
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&dLayout, dataType, m, n, n)); // ldD // output D
    {
        cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(dLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    }

    // Preference / heuristic
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t maxWs = 1 << 20; // MB
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt, opDesc, aLayout, bLayout, cLayout, dLayout,
        pref, 1, &heuristic, &returned));
    TORCH_CHECK(returned > 0, "No suitable cuBLASLt matmul algorithm found");

    void* workspace = nullptr;
    if (heuristic.workspaceSize > 0) {
        auto ce = cudaMalloc(&workspace, heuristic.workspaceSize);
        TORCH_CHECK(ce == cudaSuccess, "cudaMalloc workspace failed");
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const void* Cptr  = nullptr; // we use epilogue to do bias addition because of efficiency, we dont need to allocate C[M, N] buffer
    
    CUBLASLT_CHECK(cublasLtMatmul(
        lt, opDesc,
        &alpha,
        A_.data_ptr(), aLayout,
        B_.data_ptr(), bLayout,
        &beta,
        Cptr, cLayout,          // C input
        D.data_ptr(), dLayout,  // D output
        &heuristic.algo,
        workspace, heuristic.workspaceSize,
        stream));

    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(dLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatmulDescDestroy(opDesc);

    return D;
}

}

TORCH_LIBRARY_IMPL(xops, CUDA, m) {
  m.impl("cublaslt_mm", xops::cublaslt_mm_impl);
}