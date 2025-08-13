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
    if (t == at::kFloat)     return CUBLAS_COMPUTE_32F_FAST_TF32; // default pytorch precision
    // BF16 inputs, use fast 16bf (Tensor Cores) with FP32 accumulate
    if (t == at::kBFloat16)  return CUBLAS_COMPUTE_32F_FAST_16BF; // default pytorch precision
    TORCH_CHECK(false, "Only torch.float32 and torch.bfloat16 are supported");
}

/*
Take note because it is confusing.

* PyTorch, W[oc, ic], b[oc], Y[n, ic], Y = X @ W.T + b --- (1)

* cublasltmatmul D = alpha * (op(A) @ op(B)) + beta * C + epilogue

    We don't use beta*C for bias addition because it requires C buffer of 2-D like D
    we use epilogue. alpha = 1. Therefore, 
    
    D = alpha * (op(A) @ op(B)) + epilogue --- (2)

    Here the complexity comes, length of epilogue bias must equals #rows of D. 
    If we attempt to map (2) to (1) directly, it fails because
    D = Y, D[n, oc], N rows of D, epilogue bias is expected to n instead the actual oc we need. 

* Working backward, if we need D to have oc rows (so that we get epilogue of length oc)
    D[oc, n] = W[oc, ic] @ X.T[oc, n] + b[oc]

    let's pass W, X, b from torch.
    W row major
    X row major and is transposed on the fly
    D - if laid out row major, we need to transposed to return. If laid out col major, row major read is actually transposed operation.
        D column-major, which aliases Y.T row-major

    n=M, oc=N, ic=K
    W[oc,ic]=A[N, k]
    X[n, ic]=B[M, k]
    Y[n, oc]=D[N, M].T

    you see, variable translation is error prone, gonna stick with X, W, b
*/
at::Tensor cublaslt_linear(const at::Tensor& W, const at::Tensor& X, const c10::optional<at::Tensor>& bias) {

    CHECK_CUDA(W); CHECK_CUDA(X);
    TORCH_CHECK(W.device() == X.device(), "W and X must be on the same CUDA device");

    const auto dtype = W.scalar_type();
    TORCH_CHECK((dtype == at::kFloat || dtype == at::kBFloat16), "Only supports float32 or bfloat16; but found ", dtype);
    TORCH_CHECK(W.scalar_type() == dtype && X.scalar_type() == dtype, "W and X must be of the same dtype");

    TORCH_CHECK(W.dim() == 2 && X.dim() == 2, "W and X must be 2-D (matrix)");
    const auto OC = W.size(0);
    const auto IC = W.size(1);
    const auto N = X.size(0);
    TORCH_CHECK(X.size(1) == IC, "matmul inner dims must match: found W(", OC, ",", IC, "**) vs X(", N, ",:", X.size(1), "**)");

    at::Tensor W_ = W.contiguous();
    at::Tensor X_ = X.contiguous();
    at::Tensor bias_;
    
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA if provided");
        TORCH_CHECK(bias->device() == W.device(), "bias must be on same device as W and X");
        TORCH_CHECK(bias->scalar_type() == dtype, "bias must match dtype of W and X");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == OC,
                    "bias is expected to be 1D of length ", OC, ", but got shape ", bias->sizes());
        bias_ = bias->contiguous();  
    }

    auto Y = at::empty({N, OC}, W_.options());  // output, Y[N, OC] row-major, will be aliased to layoutD

    // Device guard & stream
    c10::cuda::CUDAGuard guard(W_.device());
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
        cublasOperation_t transW = CUBLAS_OP_N; 
        cublasOperation_t transX = CUBLAS_OP_T; // X.T
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transW, sizeof(transW)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transX, sizeof(transX)));
    }

    if (bias) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

        void* bias_ptr = bias_.data_ptr();
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }


    cublasLtMatrixLayout_t aLayout, bLayout, cLayout, dLayout;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t col_major = CUBLASLT_ORDER_COL;

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, dataType, OC, IC, IC)); // ldA, W[OC, IC] row-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bLayout, dataType,  N, IC, IC)); // ldB, X[ N, IC] row-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, dataType, OC,  N, OC)); // ldC  C unused, follow D but will be the same as D
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&dLayout, dataType, OC,  N, OC)); // ldD  Y.T[OC,N], col-major

    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_major, sizeof(col_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(dLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_major, sizeof(col_major)));

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
        W_.data_ptr(), aLayout,
        X_.data_ptr(), bLayout,
        &beta,
        Cptr, cLayout,
        Y.data_ptr(), dLayout,  // D is [OC,N] COL-major aliasing Y[N, OC] row major
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

    return Y;
}

}

TORCH_LIBRARY_IMPL(xops, CUDA, m) {
  m.impl("cublaslt_linear", xops::cublaslt_linear);
}