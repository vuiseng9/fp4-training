#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cublasLt.h>
#include <cuda_fp4.h>

#define NV_VEC_SIZE 16

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
            // TORCH_WARN("cuBLASLt heuristics cache capacity set to ", desired,
            //            " (was ", current, ")");
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
        case at::ScalarType::Byte:     return {CUDA_R_16BF, CUDA_R_4F_E2M1}; 
        default:
            TORCH_CHECK(false, "Only torch.float32, torch.bfloat16 and torch.uint8 (will be mapped to e4m3) are supported for Dout; but found ", t);
    }
}




template <typename T>
struct Pack2;

// float32 specialization
template <>
struct Pack2<float> {
  __device__ static __nv_fp4x2_storage_t
  convert(const float* row_ptr, int c0,
          __nv_fp4_interpretation_t kind, cudaRoundMode rmode)
  {
    // load two floats
    float2 f2 = make_float2(row_ptr[c0 + 0], row_ptr[c0 + 1]);
    // pack to FP4x2
    return __nv_cvt_float2_to_fp4x2(f2, kind, rmode);
  }
};

// bfloat16 specialization
template <>
struct Pack2<__nv_bfloat16> {
  __device__ static __nv_fp4x2_storage_t
  convert(const __nv_bfloat16* row_ptr, int c0,
          __nv_fp4_interpretation_t kind, cudaRoundMode rmode)
  {
    // Load two bf16 as a vector. We assume N is even so c0 is even-step.
    // If you're worried about strict-aliasing/alignment, use memcpy into __nv_bfloat162.
    __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(row_ptr + c0);

    // Convert to raw representation required by the intrinsic
    __nv_bfloat162_raw b2_raw = reinterpret_cast<const __nv_bfloat162_raw&>(b2);

    // Direct BF16(raw) -> FP4x2 conversion
    return __nv_cvt_bfloat16raw2_to_fp4x2(b2_raw, kind, rmode);
  }
};

template <typename T>
__global__ void pack_rowwise_to_2xfp4(
    const T* __restrict__ in,   // [nrow, ncol]
    uint8_t* __restrict__ out,  // [nrow, ncol/2] bytes (each byte = 2 fp4)
    int nrow, int ncol, 
    __nv_fp4_interpretation_t kind, 
    cudaRoundMode rmode)
{
  int row = blockIdx.y;
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrow) return;

  int pairs_per_row = ncol >> 1;
  if (pair_idx >= pairs_per_row) return;

  int c0 = pair_idx << 1;                   
  const T* row_ptr = in + row * ncol;       

  __nv_fp4x2_storage_t packed = Pack2<T>::convert(row_ptr, c0, kind, rmode);

  out[row * pairs_per_row + pair_idx] = reinterpret_cast<uint8_t&>(packed);
}

inline unsigned bit_floor_u32(unsigned x) {
    if (x == 0) return 0;
    return 1u << (31 - __builtin_clz(x));  // host builtin
}

inline dim3 make_block(unsigned pairs_per_row) {
    if (pairs_per_row == 0) return dim3(1,1,1);
    unsigned bx = std::min(256u, std::max(1u, bit_floor_u32(pairs_per_row)));
    return dim3(bx, 1, 1);
}

inline dim3 make_grid(int nrow, int pairs_per_row, dim3 block) {
    int gx = (pairs_per_row + block.x - 1) / block.x;
    return dim3(gx, nrow, 1);
}

std::tuple<at::Tensor, at::Tensor> cublaslt_mm_nvfp4(
    const int64_t mode, 
    const at::ScalarType DoutType,
    const at::Tensor& A, const at::Tensor& scaleA,
    const at::Tensor& B, const at::Tensor& scaleB,
    const c10::optional<at::Tensor>& bias) {
    // A and B are post-quantized data in torch.float32 or bfloat32 which will be casted to fp4 and byte-packed internally here.
    // A and B are row-major  will be treated as col-major by cublaslt.
    // scaleA and scaleB are row-major uint8 tensor and must have been sizzled. they are reinterpret as e4m3.

    // mode value 0 for forward gemm           cublaslt layout interpretation: TN (both col-m) from pytorch, A=W, B=X  (both row-m)
    // IMPORTANT: FP4 only supports TN layout in cublaslt, so is CUTLASS. We raise error but keeping the interface.
    // mode value 1 for backward gemm gradX    cublaslt layout interpretation: NN (both col-m) from pytorch, A=W, B=dY (both row-m)
    // mode value 2 for backward gemm gradW    cublaslt layout interpretation: NT (both col-m) from pytorch, A=X, B=dY (both row-m)

    // Supports Dout: fp32 or bf16 or e2m1

    TORCH_CHECK(mode == 0, "NVFP4 limitation: Only TN layout is supported");
    CHECK_CUDA(A); CHECK_CUDA(B); CHECK_CUDA(scaleA); CHECK_CUDA(scaleB);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
    TORCH_CHECK(scaleA.device() == scaleB.device(), "scaleA and scaleB must be on the same CUDA device");
    TORCH_CHECK(A.device() == scaleA.device(), "A, B, scaleA and scaleB must be on the same CUDA device");
    CHECK_2D(A); CHECK_2D(B); CHECK_2D(scaleA); CHECK_2D(scaleB);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(scaleA); CHECK_CONTIGUOUS(scaleB);
    // TODO!!! check dimension of scaleA & scaleB
    
    // Different to MXFP8, A & B won't be in fp4 yet, we will pack here, e.g. 2 float32/bf16 to 1 byte made up of 2 fp4
    // therefore, we check against pre-quantized type. note that the values are in FP4
    const auto el_type = A.scalar_type();
    TORCH_CHECK((el_type == at::ScalarType::Float || el_type == at::ScalarType::BFloat16), "Only supports torch.float32/bfloat16 (will be cast to e2m1 and pack 2x into a byte of e4m3) for A; but found ", el_type);
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must be of the same dtype");
    
    // Scaling Factor Type of NVFP4 is e4m3.
    const auto scale_type = scaleA.scalar_type();
    TORCH_CHECK((scale_type == c10::ScalarType::Byte), "Only supports torch.uint8 (will be reinterpreted as e4m3) for scaleA; but found ", scale_type);
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

    // TODO(fp4-last)
    if (DoutType == at::kByte) {  // when dout it uint8 alias to 2x e2m1
        TORCH_CHECK(
            (m % NV_VEC_SIZE) == 0,
            "When Dout is e2m1 (stored as uint8), batch size m must be a multiple of 16; got m=", m, "."
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

    // Packing Tensor A and B ----------------------------------------------------------------------------------
    TORCH_CHECK((k % 2) == 0, "k must be even (pad horizontally if needed).");
    int64_t pairs_per_row = k >> 1;
    at::Tensor packedA = at::empty({m, pairs_per_row}, A.options().dtype(at::kByte));
    dim3 block = make_block(pairs_per_row);
    dim3 grid  = make_grid(m, pairs_per_row, block);
    
    if (el_type == at::ScalarType::Float) {
        pack_rowwise_to_2xfp4<<<grid, block, 0, stream>>>(
            A.data_ptr<float>(), packedA.data_ptr<uint8_t>(),
            (int)m, (int)k, __NV_E2M1, cudaRoundNearest
        );
    } else {
        pack_rowwise_to_2xfp4<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(B.data_ptr<at::BFloat16>()), 
            packedA.data_ptr<uint8_t>(),
            (int)m, (int)k, __NV_E2M1, cudaRoundNearest
        );
    }

    at::Tensor packedB = at::empty({n, pairs_per_row}, B.options().dtype(at::kByte));
    grid  = make_grid(n, pairs_per_row, block);

    if (el_type == at::ScalarType::Float) {
        pack_rowwise_to_2xfp4<<<grid, block, 0, stream>>>(
            B.data_ptr<float>(), packedB.data_ptr<uint8_t>(),
            (int)n, (int)k, __NV_E2M1, cudaRoundNearest
        );
    } else {
        pack_rowwise_to_2xfp4<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(B.data_ptr<at::BFloat16>()), 
            packedB.data_ptr<uint8_t>(),
            (int)n, (int)k, __NV_E2M1, cudaRoundNearest
        );
    }

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

    // nvfp4 A/B scaling factors are of fp8_e4m3, C/D can be fp32 scalar 
    cublasLtMatmulMatrixScale_t scaleType_e4m3 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scaleType_f32  = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    // A, B, C, Dout layout descriptor 
    // use default column major; therefore no need to set CUBLASLT_MATRIX_LAYOUT_ORDER
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DoutLayout;

    // A matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    cublasOperation_t transA = isTransA ? CUBLAS_OP_T : CUBLAS_OP_N; 
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_4F_E2M1, isTransA? k:m, isTransA? m:k, isTransA? k:m)); // A[m,k] ; A.T[k,m]

    __nv_fp8_e4m3 *scaleA_ptr = static_cast<__nv_fp8_e4m3 *>(scaleA.data_ptr());
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,    &scaleType_e4m3, sizeof(scaleType_e4m3)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA_ptr,     sizeof(scaleA_ptr)));
    
    // B matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    cublasOperation_t transB = isTransB ? CUBLAS_OP_T : CUBLAS_OP_N; 
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_4F_E2M1, isTransB? n:k, isTransB? k:n, isTransB? n:k)); // B[k,n] ; B.T[n,k]

    __nv_fp8_e4m3 *scaleB_ptr = static_cast<__nv_fp8_e4m3 *>(scaleB.data_ptr());
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,    &scaleType_e4m3, sizeof(scaleType_e4m3)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB_ptr,     sizeof(scaleB_ptr)));
    
    // C matrix and scaling factor ----------------------------------------------------------------------------------------------------------
    // C is unused for linear layer case.
    // But its configs are dependant on Dout choice.
    // e.g. Dout=fp32, C=fp32
    //      Dout=bf16, C=bf16
    //      Dout=e2m1, C=bf16
    auto [c_dtype, dout_dtype] = map_c_dout_cuda_dtype(DoutType);
    const void* Cptr  = nullptr; // we use epilogue to do bias addition because of efficiency, we dont need to allocate C[M, N] buffer
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE,     &scaleType_f32,  sizeof(scaleType_f32)));
    // CUBLASLT_MATMUL_DESC_C_SCALE_POINTER - If not specified, or set to NULL, the scaling factor is assumed to be 1
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&CLayout, c_dtype, m,  n, m)); // C unused, but must comply to D 

    // D (in) matrix and scaling factor ------------------------------------------------------------------------------------
    // CUBLASLT_MATMUL_DESC_D_SCALE_MODE is only used for NVFP4 
    // AND only when output is FP4. Forbidden for other output type  
    // CUBLASLT_MATMUL_DESC_D_SCALE_POINTER - If not specified, or set to NULL, the scaling factor is assumed to be 1
    // float scaleDin = 1.0f;
    // float *scaleDin_ptr = &scaleDin; 
    // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE,    &scaleType_f32,  sizeof(scaleType_f32)));
    // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleDin_ptr,   sizeof(scaleDin_ptr)));

    // D out matrix and scaling factor ------------------------------------------------------------------------------------------------------
    at::Tensor scaleDout;
    if (dout_dtype == CUDA_R_4F_E2M1) {
        // scaleDout must fit to tile size 4x128 (To verify on layout, unswizzle)
        scaleDout = at::empty_strided({ roundoff(m/32, 4), roundoff(n, 128) }, {n, 1}, scaleA.options()); // force contiguous row-major although scaleA is row-major as well
        __nv_fp8_e4m3 *scaleDout_ptr = static_cast<__nv_fp8_e4m3 *>(scaleDout.data_ptr());
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,    &scaleType_e4m3, sizeof(scaleType_e4m3)));
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
    // TODO(fp4-last) - how do deal with this nvfp4

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
        packedA.data_ptr(), ALayout,
        packedB.data_ptr(), BLayout,
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
  m.impl("cublaslt_mm_nvfp4", xops::cublaslt_mm_nvfp4);
}