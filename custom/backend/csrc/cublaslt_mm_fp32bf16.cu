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
#define CHECK_2D(x) TORCH_CHECK(x.dim() == 2, #x " must be a 2D tensor")


namespace xops {

inline cudaDataType to_cuda_dtype(at::ScalarType t) {
    if (t == at::kFloat)     return CUDA_R_32F;
    if (t == at::kBFloat16)  return CUDA_R_16BF;
    TORCH_CHECK(false, "Only torch.float32 and torch.bfloat16 are supported");
}

inline cublasComputeType_t to_compute_type(at::ScalarType t) {
    if (t == at::kFloat)     return CUBLAS_COMPUTE_32F; // default pytorch precision
    // BF16 inputs, use fast 16bf (Tensor Cores) with FP32 accumulate
    if (t == at::kBFloat16)  return CUBLAS_COMPUTE_16F; // default pytorch precision
    TORCH_CHECK(false, "Only torch.float32 and torch.bfloat16 are supported");
}

// One-time init of cuBLASLt heuristics cache capacity (thread-safe)
static void ensure_cublaslt_cache_configured() {
    static std::once_flag once;
    std::call_once(once, [](){
        size_t current = 0;
        cublasStatus_t st = cublasLtHeuristicsCacheGetCapacity(&current);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                    "cublasLtHeuristicsCacheGetCapacity failed");

        const size_t desired = static_cast<size_t>(32768);
        if (current != desired) {
            st = cublasLtHeuristicsCacheSetCapacity(desired);
            TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                        "cublasLtHeuristicsCacheSetCapacity(", desired,
                        ") failed (current was ", current, ")");
            // Optional: uncomment to see a one-time note
            TORCH_WARN("cuBLASLt heuristics cache capacity set to ", desired,
                       " (was ", current, ")");
        }
        // else: already at desired capacity; nothing to do.
    });
}

inline bool is_A_transposed(int mode) {
    switch (mode) {
        case 0: return true;   // TN (fwd)
        case 1: return false;  // NN (xgrad)
        case 2: return false;  // NT (wgrad)
        default:
            throw std::out_of_range("is_A_transposed: invalid mode (expected 0,1,2)");
    }
}

inline bool is_B_transposed(int mode) {
    switch (mode) {
        case 0: return false; // TN (fwd)
        case 1: return false; // NN (xgrad)
        case 2: return true;  // NT (wgrad)
        default:
            throw std::out_of_range("is_B_transposed: invalid mode (expected 0,1,2)");
    }
}

at::Tensor cublaslt_mm_fp32_or_bf16(const int64_t mode, at::Tensor& A, const at::Tensor& B, const c10::optional<at::Tensor>& bias) {
    // mode value 0 for forward gemm           cublaslt layout interpretation: TN (both col-m) from pytorch, A=W, B=X  (both row-m)
    // mode value 1 for backward gemm gradX    cublaslt layout interpretation: NN (both col-m) from pytorch, A=W, B=dY (both row-m)
    // mode value 2 for backward gemm gradW    cublaslt layout interpretation: NT (both col-m) from pytorch, A=X, B=dY (both row-m)

    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
    CHECK_2D(A); CHECK_2D(B);
    
    const auto el_type = A.scalar_type();
    TORCH_CHECK((el_type == at::kFloat || el_type == at::kBFloat16), "This op only supports float32 or bfloat16; but found ", el_type);
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must be of the same dtype");
    
    const bool isTransA = is_A_transposed(mode);
    const bool isTransB = is_B_transposed(mode);

    // There is quite a complex mapping
    // A=W[oc, ic] row major in pytorch
    // [ic, oc] col major interpreted by cublast
    // require transpose for matmul, meaning [oc, ic]
    // m=oc=A.size(0); k=ic=A.size(1) if transposed.
    // remember A is still torch.tensor, if we want oc, we need to get the oc dim in torch tensor.
    const auto m  = isTransA ? A.size(0) : A.size(1);
    const auto k  = isTransA ? A.size(1) : A.size(0);
    // B=X[n, ic] row major in pytorch
    // [ic, n] col major intepreted by cublaslt
    // since this is ready for matmul
    // k=ic=B.size(1), n=n=B.size(0) if no transpose
    // remember B is still torch.tensor, if we want ic, we need to get the oc dim in torch tensor.
    const auto kB = isTransB ? B.size(0) : B.size(1);
    const auto n  = isTransB ? B.size(1) : B.size(0);
    TORCH_CHECK(k == kB, "Mismatch matmul inner: A's k = ", k, ", B's k =", kB);

    at::Tensor bias_;
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be on CUDA");
        TORCH_CHECK(bias->device() == A.device(), "bias must be on same device as A and B");
        TORCH_CHECK(bias->scalar_type() == el_type, "bias must match element type of A and B but found ", bias->scalar_type());
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == m, "bias is expected to be 1D of length ", m, ", but got shape ", bias->sizes());
        bias_ = bias->contiguous();
    }
    
    // Device guard & stream -----------------------------------------------------------------------------------
    c10::cuda::CUDAGuard guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use PyTorch's pooled cuBLASLt handle (don't create/destroy yourself)
    cublasLtHandle_t lt = at::cuda::getCurrentCUDABlasLtHandle();
    
    // Ensure cuBLASLt's internal heuristics cache is configured
    ensure_cublaslt_cache_configured();

    // Matmul descriptor ----------------------------------------------------------------------
    const cudaDataType        dataType = to_cuda_dtype(el_type);
    const cublasComputeType_t computeType = to_compute_type(el_type);
    const cudaDataType        scaleType = CUDA_R_32F;  // alpha/beta in float

    cublasLtMatmulDesc_t opDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
    
    cublasOperation_t transA = isTransA ? CUBLAS_OP_T : CUBLAS_OP_N; 
    cublasOperation_t transB = isTransB ? CUBLAS_OP_T : CUBLAS_OP_N; 
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Epilogue  -----------------------------------------------------------------------------
    cublasLtEpilogue_t epi = bias.has_value() ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

    if (bias.has_value()) {
        void* bias_ptr = bias_.data_ptr();
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }

    // A, B, C, D layout descriptor -----------------------------------------------------------------------------
    // use default column major; therefore no need to set CUBLASLT_MATRIX_LAYOUT_ORDER
    cublasLtMatrixLayout_t aLayout, bLayout, cLayout, dLayout;

    // cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, dataType, isTransA? k : m, isTransA? m : k, isTransA? k : m)); // A[m,k] ; A.T[k,m]
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bLayout, dataType, isTransB? n : k, isTransB? k : n, isTransB? n : k)); // B[k,n] ; B.T[n,k]
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, dataType, m,  n, m)); // C unused, but must comply to D 
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&dLayout, dataType, m,  n, m)); // D[m,n]
    
    // Define Y torch tensor output -----------------------------------------------------------------------------
    // Important this is pytorch object, Y[n=n, m=oc] row_major.
    // cublaslt output will be D[m=oc, n=n] col_major, we just assign data pointer to Y tensor
    // denote with Dt instead of Y since this function will be used for other gemm in linear
    auto Dt = at::empty_strided({n, m}, {m, 1}, A.options()); // Explicit strides (always row-major)
    // notice that D[m, n] col-m, Y[n, m] row-m

    // Preference / heuristic -----------------------------------------------------------------------------------
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t maxWs = 1 << 20; // 1MiB
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
        A.data_ptr(), aLayout,
        B.data_ptr(), bLayout,
        &beta,
        Cptr, cLayout,
        Dt.data_ptr(), dLayout,
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

    return Dt;
}
}

TORCH_LIBRARY_IMPL(xops, CUDA, m) {
  m.impl("cublaslt_mm_fp32_or_bf16", xops::cublaslt_mm_fp32_or_bf16);
}