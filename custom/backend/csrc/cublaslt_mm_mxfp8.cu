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
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace xops {

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

inline int64_t roundoff(int64_t  x, int64_t granul) {
    return granul * ((x + (granul - 1)) / granul);
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

using c_dout_dtype = std::tuple<cudaDataType_t, cudaDataType_t>;

inline c_dout_dtype map_c_dout_cuda_dtype(at::ScalarType t) {
    // return CType and DoutType
    switch (t) {
        case at::ScalarType::Float:    return {CUDA_R_32F,  CUDA_R_32F};
        case at::ScalarType::BFloat16: return {CUDA_R_16BF, CUDA_R_16BF};
        case at::ScalarType::Byte:     return {CUDA_R_16BF, CUDA_R_8F_E4M3}; 
        default:
            TORCH_CHECK(false, "Only torch.float32, torch.bfloat16 and torch.uint8 (will be mapped to e4m3) are supported for Dout; but found ", t);
    }
}

std::tuple<at::Tensor, at::Tensor> cublaslt_mm_mxfp8(
    const int64_t mode, 
    const at::ScalarType DoutType,
    const at::Tensor& A, const at::Tensor& scaleA,
    const at::Tensor& B, const at::Tensor& scaleB,
    const c10::optional<at::Tensor>& bias) {
    // A and B are row major torch.float8_e4m3fn will be treated as col-major by cublaslt.
    // mode value 0 for forward gemm           cublaslt layout interpretation: TN (both col-m) from pytorch, A=W, B=X  (both row-m)
    // mode value 1 for backward gemm gradX    cublaslt layout interpretation: NN (both col-m) from pytorch, A=W, B=dY (both row-m)
    // mode value 2 for backward gemm gradW    cublaslt layout interpretation: NT (both col-m) from pytorch, A=X, B=dY (both row-m)
    // scaleA and scaleB are row-major uint8 tensor and must be sizzled. they are reinterpret as e8m0. 

    // Supports Dout/C=fp32
    //          Dout/C=fp32 or bf16
    //          Dout=e4m3 C=bf16

    CHECK_CUDA(A); CHECK_CUDA(B); CHECK_CUDA(scaleA); CHECK_CUDA(scaleB);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
    TORCH_CHECK(scaleA.device() == scaleB.device(), "scaleA and scaleB must be on the same CUDA device");
    TORCH_CHECK(A.device() == scaleA.device(), "A, B, scaleA and scaleB must be on the same CUDA device");
    CHECK_2D(A); CHECK_2D(B); CHECK_2D(scaleA); CHECK_2D(scaleB);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(scaleA); CHECK_CONTIGUOUS(scaleB);
    // TODO!!! check dimension of scaleA & scaleB
    
    const auto el_type = A.scalar_type();
    TORCH_CHECK((el_type == c10::ScalarType::Byte), "Only supports torch.uint8 (will be reinterpreted as e4m3) for A; but found ", el_type);
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must be of the same dtype");
    
    const auto scale_type = scaleA.scalar_type();
    TORCH_CHECK((scale_type == c10::ScalarType::Byte), "Only supports torch.uint8 (will be reinterpreted as e8m0) for scaleA; but found ", scale_type);
    TORCH_CHECK(scaleA.scalar_type() == scaleB.scalar_type(), "scaleA and scaleB must be of the same dtype");

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

    if (DoutType == at::kByte) {  // when dout it uint8 alias to f8_e4m3
        TORCH_CHECK(
            (m % 32) == 0,
            "When Dout is FP8 (stored as uint8), batch size m must be a multiple of 32; got m=",
            m, "."
        );
    }

    at::Tensor bias_;
    at::ScalarType bias_dtype;

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be on CUDA");
        TORCH_CHECK(bias->device() == A.device(), "bias must be on same device as A and B");
        bias_dtype = bias->scalar_type();
        TORCH_CHECK(bias_dtype == at::kFloat || bias_dtype == at::kBFloat16, "Only supports bias of torch.float32/bfloat16, found ", bias->scalar_type());
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == m, "bias is expected to be 1D of length ", m, ", but got shape ", bias->sizes());

        // cublaslt only supports BF/FP16 bias epilogue but we only implement for BF16
        if (bias_dtype == at::kFloat) {
            bias_ = bias->contiguous().to(at::kBFloat16);  // FP32 -> BF16
            // .to() return a new copy, therefore no need to restore bias.
        } else {
            bias_ = bias->contiguous();                    // already BF16
        }
    }
    
    // Device guard & stream -----------------------------------------------------------------------------------
    c10::cuda::CUDAGuard guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use PyTorch's pooled cuBLASLt handle (don't create/destroy yourself)
    cublasLtHandle_t lt = at::cuda::getCurrentCUDABlasLtHandle();
    
    // Ensure cuBLASLt's internal heuristics cache is configured
    ensure_cublaslt_cache_configured();

    // Matmul descriptor ----------------------------------------------------------------------
    
    cublasLtMatmulDesc_t opDesc;
    const cublasComputeType_t computeType = CUBLAS_COMPUTE_32F; // no other compute types are supported for MXFP8/NVFP4
    const cudaDataType        scaleType   = CUDA_R_32F;         // alpha/beta in float
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
    // how to organize the attribute setup? look at canonical cublaslt gemm, we setup for each input, type, layout

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // CUBLASLT_MATMUL_DESC_SCALE_TYPE is for alpha and beta, default value depends on CUBLASLT_MATMUL_DESC_COMPUTE_TYPE. 
    // Keeping default type for now. opDesc has initialized a scale type.

    // mxfp8 A/B scaling factors are of fp8_e8m0, C/D can be fp32 scalar 
    cublasLtMatmulMatrixScale_t scaleType_e8m0 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulMatrixScale_t scaleType_f32  = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    // A, B, C, Dout layout descriptor 
    // use default column major; therefore no need to set CUBLASLT_MATRIX_LAYOUT_ORDER
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DoutLayout;

    // A matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    cublasOperation_t transA = isTransA ? CUBLAS_OP_T : CUBLAS_OP_N; 
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_8F_E4M3, isTransA? k:m, isTransA? m:k, isTransA? k:m)); // A[m,k] ; A.T[k,m]
    
    __nv_fp8_e8m0 *scaleA_ptr = static_cast<__nv_fp8_e8m0 *>(scaleA.data_ptr());
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,    &scaleType_e8m0, sizeof(scaleType_e8m0)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA_ptr,     sizeof(scaleA_ptr)));
    
    // B matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    cublasOperation_t transB = isTransB ? CUBLAS_OP_T : CUBLAS_OP_N; 
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_8F_E4M3, isTransB? n:k, isTransB? k:n, isTransB? n:k)); // B[k,n] ; B.T[n,k]
    
    __nv_fp8_e8m0 *scaleB_ptr = static_cast<__nv_fp8_e8m0 *>(scaleB.data_ptr());
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,    &scaleType_e8m0, sizeof(scaleType_e8m0)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB_ptr,     sizeof(scaleB_ptr)));
    
    // C matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    // C is unused for linear layer case.
    // But its configs are dependant on Dout choice.
    // e.g. Dout=fp32, C=fp32
    //      Dout=bf16, C=bf16
    //      Dout=e4m3, C=bf16
    auto [c_dtype, dout_dtype] = map_c_dout_cuda_dtype(DoutType);
    const void* Cptr  = nullptr; // we use epilogue to do bias addition because of efficiency, we dont need to allocate C[M, N] buffer
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE,     &scaleType_f32,  sizeof(scaleType_f32)));
    // CUBLASLT_MATMUL_DESC_C_SCALE_POINTER - If not specified, or set to NULL, the scaling factor is assumed to be 1
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&CLayout, c_dtype, m,  n, m)); // C unused, but must comply to D 

    // [Not applicable, for nvfp4] D (in) matrix and scaling factor ------------------------------------------------------------------------------------
    // [Not applicable, for nvfp4] CUBLASLT_MATMUL_DESC_D_SCALE_MODE is only used for NVFP4, i.e. when A/B_SCALE_MODE in VEC16_UE4M3 
    // [Not applicable, for nvfp4] CUBLASLT_MATMUL_DESC_D_SCALE_POINTER - If not specified, or set to NULL, the scaling factor is assumed to be 1
    
    // D out matrix and scaling factor ------------------------------------------------------------------------------------------------------
    at::Tensor scaleDout;
    if (dout_dtype == CUDA_R_8F_E4M3) {
        // scaleDout must fit to tile size 4x128 (To verify on layout, unswizzle)
        scaleDout = at::empty_strided({ roundoff(m/32, 4), roundoff(n, 128) }, {n, 1}, scaleA.options()); // force contiguous row-major although scaleA is row-major as well
        __nv_fp8_e8m0 *scaleDout_ptr = static_cast<__nv_fp8_e8m0 *>(scaleDout.data_ptr());
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,    &scaleType_e8m0, sizeof(scaleType_e8m0)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &scaleDout_ptr,  sizeof(scaleDout_ptr)));
    } else {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleType_f32,  sizeof(scaleType_f32)));
    }
    // CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER ? keep default for now.
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&DoutLayout, dout_dtype, m,  n, m)); // D[m,n]

    // Define Y torch tensor output -----------------------------------------------------------------------------
    // Important: this is a pytorch tensor it will be interpreted by pytorch as row-majored Y[n=n, m=oc].
    // cublaslt output will be D[m=oc, n=n] col_major, torch needs the transpose of it.
    // denote with Dt instead of Y for clarity that it is a transpose of cublaslt output
    // still confused? Notice that Dt[n, m] row-m, outLayout is [m, n] col-m
    auto Dt = at::empty_strided({n, m}, {m, 1}, A.options().dtype(DoutType)); // Explicit strides (always row-major)

    // Epilogue ----------------------------------------------------------------------------------------------
    cublasLtEpilogue_t epi = bias.has_value() ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));

    if (bias.has_value()) {
        void* bias_ptr = bias_.data_ptr();
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }

    // Preference / heuristic to find gemm algorithm ---------------------------------------------------------
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t maxWs = 1 << 20; // 1MiB
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt, opDesc, ALayout, BLayout, CLayout, DoutLayout,
        pref, 1, &heuristic, &returned));
    TORCH_CHECK(returned > 0, "No suitable cuBLASLt matmul algorithm found");

    void* workspace = nullptr;
    if (heuristic.workspaceSize > 0) {
        auto ce = cudaMalloc(&workspace, heuristic.workspaceSize);
        TORCH_CHECK(ce == cudaSuccess, "cudaMalloc workspace failed");
    }

    // Run Matmul
    CUBLASLT_CHECK(cublasLtMatmul(
        lt, opDesc,
        &alpha,
        A.data_ptr(), ALayout,
        B.data_ptr(), BLayout,
        &beta,
        Cptr, CLayout,
        Dt.data_ptr(), DoutLayout,
        &heuristic.algo,
        workspace, heuristic.workspaceSize,
        stream));

    // Cleanup
    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(DoutLayout);
    cublasLtMatrixLayoutDestroy(CLayout);
    cublasLtMatrixLayoutDestroy(BLayout);
    cublasLtMatrixLayoutDestroy(ALayout);
    cublasLtMatmulDescDestroy(opDesc);

    return {Dt, scaleDout};
}
}

TORCH_LIBRARY_IMPL(xops, CUDA, m) {
  m.impl("cublaslt_mm_mxfp8", xops::cublaslt_mm_mxfp8);
}