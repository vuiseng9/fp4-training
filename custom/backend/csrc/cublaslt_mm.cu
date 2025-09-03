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
    if (t == at::kFloat)     return CUBLAS_COMPUTE_32F_FAST_TF32; // default pytorch precision
    // BF16 inputs, use fast 16bf (Tensor Cores) with FP32 accumulate
    if (t == at::kBFloat16)  return CUBLAS_COMPUTE_32F_FAST_16BF; // default pytorch precision
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

/*
Take note because it is confusing.

* PyTorch, W[oc, ic], b[oc], Y[n, ic], Y = X @ W.T + b --- (1)

* cublasltmatmul D = alpha * (op(A) @ op(B)) + beta * C + epilogue

    We don't use beta*C for bias addition because it requires C buffer of 2-D like D
    we use epilogue. alpha = 1. Therefore, 
    
    _D = alpha * (op(A) @ op(B)) --- (2)
    D  = epilogue(_D)

    Here the complexity comes, length of epilogue bias must equals #rows of D/_D. 
    If we attempt to map (2) to (1) directly, it is invalid because
    Y[n, oc] has n rows, bias must be of length OC, meaning D must be [oc, n].

* Working backward, if we need D to have oc rows (so that we get epilogue of length oc)
    _D[oc, n] = A[oc, ic] @ B[ic, n]
    
    Y.T[oc, n] = W[oc, ic] @ X.T[ic, n]

    cublaslt uses col-major layout, torch user row-major layout
    
    A: pass W[oc, ic] row-major, at cublast level op(W[oc, ic] col-major) where op() is layout transformation to col-major

    B: if X.T, X.T[ic, n] is actually in col-major, we just pass X.T to cublaslt, not op() needed.
    
    D: output in D[oc, n] is in col-major but it can be interprete as D.T[n, oc] row-major. we just point the Y data pointer to D data pointer
    Configure Y to be [n, oc] row-major

    mapping to M=oc, K=ic ,N=n

*/
at::Tensor cublaslt_linear_fwd(const at::Tensor& row_major_W, const at::Tensor& col_major_Xt, const c10::optional<at::Tensor>& bias) {
    // A: W[oc, ic] in row-major
    // B: X.T[ic, n] in col-major

    CHECK_CUDA(row_major_W); CHECK_CUDA(col_major_Xt);
    TORCH_CHECK(row_major_W.device() == col_major_Xt.device(), "W and Xt must be on the same CUDA device");

    CHECK_2D(row_major_W); CHECK_2D(col_major_Xt);
    TORCH_CHECK(row_major_W.is_contiguous(), "W must be row-major");
    TORCH_CHECK(col_major_Xt.is_contiguous() == false, "Xt must be column-major");

    const auto el_type = row_major_W.scalar_type();
    TORCH_CHECK((el_type == at::kFloat || el_type == at::kBFloat16), "Only supports float32 or bfloat16; but found ", el_type);
    TORCH_CHECK(row_major_W.scalar_type() == el_type && col_major_Xt.scalar_type() == el_type, "W and Xt must be of the same dtype");

    const auto oc = row_major_W.size(0);
    const auto ic = row_major_W.size(1);
    const auto n = col_major_Xt.size(1);
    TORCH_CHECK(col_major_Xt.size(0) == ic, "matmul inner dims must match: found X(", oc, ",", ic, "**) vs Xt(", col_major_Xt.size(1), ",:", n, "**)");

    at::Tensor bias_;
    
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA if provided");
        TORCH_CHECK(bias->device() == row_major_W.device(), "bias must be on same device as W and Xt");
        TORCH_CHECK(bias->scalar_type() == el_type, "bias must match dtype of W and Xt");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == oc,
                    "bias is expected to be 1D of length ", oc, ", but got shape ", bias->sizes());
        bias_ = bias->contiguous();
    }

    auto row_major_Y = at::empty({n, oc}, row_major_W.options());  
    // Important this is pytorch object, Y[n, oc] row_major.
    // cublaslt output will be D[oc, n] col_major, we just assign data pointer to Y tensor

    // Device guard & stream
    c10::cuda::CUDAGuard guard(row_major_W.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use PyTorch's pooled cuBLASLt handle (don't create/destroy yourself)
    cublasLtHandle_t lt = at::cuda::getCurrentCUDABlasLtHandle();

    // Ensure cuBLASLt's internal heuristics cache is configured
    ensure_cublaslt_cache_configured();

    const cudaDataType dataType = to_cuda_dtype(el_type);
    const cublasComputeType_t computeType = to_compute_type(el_type);
    const cudaDataType scaleType = CUDA_R_32F; // alpha/beta in float

    // Matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
    {
        cublasOperation_t transW = CUBLAS_OP_N; 
        cublasOperation_t transXt = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transW, sizeof(transW)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transXt, sizeof(transXt)));
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
    // cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, dataType, oc, ic, ic)); // A/  W[oc, ic] row-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bLayout, dataType, ic,  n, ic)); // B/X.T[ic,  n] col-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_major, sizeof(col_major)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, dataType, oc,  n, oc)); // C unused, but must comply with D 
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_major, sizeof(col_major)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&dLayout, dataType, oc,  n, oc)); // D/Y.T[oc,  n], col-major
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
        row_major_W.data_ptr(), aLayout,
        col_major_Xt.data_ptr(), bLayout,
        &beta,
        Cptr, cLayout,
        row_major_Y.data_ptr(), dLayout,
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

    return row_major_Y;
}

// matmul without bias, A, B are row-major, so is output
// for ease of use, less confusion of mapping tensor axes
at::Tensor cublaslt_mm_xbias(const at::Tensor& A, const at::Tensor& B) {
    // row major A and row major B
    
    CHECK_CUDA(A); CHECK_CUDA(B);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");

    const auto dtype = A.scalar_type();
    TORCH_CHECK((dtype == at::kFloat || dtype == at::kBFloat16), "Only supports float32 or bfloat16; but found ", dtype);
    TORCH_CHECK(A.scalar_type() == dtype && B.scalar_type() == dtype, "A and B must be of the same dtype");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D (matrix)");
    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "matmul inner dims must match: found A(", M, ",", K, "**) vs B(", B.size(0), ",:", N, "**)");

    at::Tensor A_ = A.contiguous();
    at::Tensor B_ = B.contiguous();

    auto D = at::empty({M, N}, A_.options()); 

    // Device guard & stream
    c10::cuda::CUDAGuard guard(A_.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasLtHandle_t lt = at::cuda::getCurrentCUDABlasLtHandle();

    // Ensure cuBLASLt's internal heuristics cache is configured
    ensure_cublaslt_cache_configured();

    const cudaDataType dataType = to_cuda_dtype(dtype);
    const cublasComputeType_t computeType = to_compute_type(dtype);
    const cudaDataType scaleType = CUDA_R_32F; // alpha/beta in float

    // Matmul descriptor
    cublasLtMatmulDesc_t opDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
    {
        cublasOperation_t transA = CUBLAS_OP_N;
        cublasOperation_t transB = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    }

    cublasLtMatrixLayout_t aLayout, bLayout, cLayout, dLayout;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aLayout, dataType, M, K, K)); // A[M, K], ldA
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bLayout, dataType, K, N, N)); // B[K, N], ldB
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cLayout, dataType, M, N, N)); // C[M, N], ldC
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&dLayout, dataType, M, N, N)); // D[M, N], ldD

    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(dLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));

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
        Cptr, cLayout,
        D.data_ptr(), dLayout,  // D is [OC,N] COL-major aliasing Y[N, OC] row major
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
  m.impl("cublaslt_matmul_bias_epilogue", xops::cublaslt_linear_fwd);
  m.impl("cublaslt_matmul_xbias", xops::cublaslt_mm_xbias);
}