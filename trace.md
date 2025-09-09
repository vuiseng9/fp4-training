


tex.quantize

tex.generic

### tex.quantize
grep -rn tex.quantize | grep "^transformer_engine/pytorch\|tex.quantize"
transformer_engine/pytorch/module/layernorm_mlp.py:435:                # tex.quantize does not support GELU fusion for blockwise.
transformer_engine/pytorch/module/layernorm_mlp.py:437:                act_out = tex.quantize(act_out, fc2_input_quantizer)
transformer_engine/pytorch/module/base.py:1353:                tex.quantize(tensor, quantizer, out, skip_update_flag)
transformer_engine/pytorch/tensor/float8_tensor.py:85:        tex.quantize(src, self, dst, noop_flag)
transformer_engine/pytorch/tensor/float8_tensor.py:243:        tex.quantize(src, self, dst, noop_flag)
transformer_engine/pytorch/tensor/quantized_tensor.py:268:        return tex.quantize(tensor, quantizer)
transformer_engine/pytorch/tensor/mxfp8_tensor.py:63:        tex.quantize(src, self, dst, noop_flag)
transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:99:        tex.quantize(src, self, dst, noop_flag)

code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_mlp.py:435
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/layernorm_mlp.py:437
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/base.py:1353
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/tensor/float8_tensor.py:85
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/tensor/float8_tensor.py:243
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/tensor/quantized_tensor.py:268
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/tensor/mxfp8_tensor.py:63
code -g /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:99

in our mxfp8 block scaling script, always tex.quantize @ tensor/quantized_tensor.py:268

### tex.generic_gemm
pytorch/cpp_extensions/gemm.py:113:    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

tex.generic_gemm is only called here, wrap in general_gemm

because it is wrapped in python by one function only, i.e. gemm.general_gemm, we can easily put a snippet to show the caller, no other path elsewhere to complicate tracing.



#### Traces for one training step (branch: caller-of-quantize-gemm)
```python
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
    class _Linear(torch.autograd.Function):
        forward 204
        input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
        inputmat = input_quantizer(inputmat)
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
    class _Linear(torch.autograd.Function):
        forward 231
        weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
        )
general_gemm @ forward in pytorch/module/linear.py:286
    class _Linear(torch.autograd.Function):
        forward 286
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
general_gemm @ forward in pytorch/module/linear.py:286
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
general_gemm @ forward in pytorch/module/linear.py:286
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
general_gemm @ forward in pytorch/module/linear.py:286
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
general_gemm @ forward in pytorch/module/linear.py:286
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
tex.quantize <- _QuantizeFunc.forward called by quantize in pytorch/tensor/quantized_tensor.py:203
general_gemm @ forward in pytorch/module/linear.py:286
tex.bgrad_quantize @ module/base.py:1143
    class TransformerEngineBaseModule:
        def grad_output_preprocess
general_gemm @ backward in pytorch/module/linear.py:622
    class _Linear(torch.autograd.Function):
        backward 662 dgrad
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
    class _Linear(torch.autograd.Function):
        backward 771 wgrad & bias grad
tex.bgrad_quantize @ module/base.py:1143
general_gemm @ backward in pytorch/module/linear.py:622
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
tex.bgrad_quantize @ module/base.py:1143
general_gemm @ backward in pytorch/module/linear.py:622
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
tex.bgrad_quantize @ module/base.py:1143
general_gemm @ backward in pytorch/module/linear.py:622
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
tex.bgrad_quantize @ module/base.py:1143
general_gemm @ backward in pytorch/module/linear.py:622
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
tex.bgrad_quantize @ module/base.py:1143
general_gemm @ backward in pytorch/module/linear.py:622
general_gemm @ wgrad_gemm in pytorch/module/linear.py:771
```

basically, the kernel quantize both direction in one go
following are for X and W
torch.float32 MXFP8Quantizer(rowwise_usage=True, columnwise_usage=True, internal=True, )
how about incoming gradient?

x and dy used when calling general_gemm is already quantized, MXFP8TensorBase
weight_fp8 and grad_output too

question is where is grad_output quantized
TransformerEngineBaseModule.grad_output_preprocess line 532 in module._Linear
tex.bgrad_quantize (hidden here!)


### signature
pytorch/tensor/quantized_tensor.py
class _QuantizeFunc(torch.autograd.Function):
    def forward
        tex.quantize(tensor, quantizer)
            tensor, f32
            MXFP8Quantizer(rowwise_usage=True, columnwise_usage=True, internal=True, )

transformer_engine/pytorch/csrc/extensions/pybind.cpp:344

class TransformerEngineBaseModule(torch.nn.Module):
    @staticmethod
    def grad_output_preprocess(
        ctx,
        grad_output: torch.Tensor,
        row_parallel_mode: bool,
        quantizer: Optional[Quantizer],
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        grad_bias, grad_output = tex.bgrad_quantize(grad_output, quantizer)
        grad_output, fp32, 
        MXFP8Quantizer(rowwise_usage=True, columnwise_usage=True, internal=True, )


Questions, do we quantize or dequantize

transformer_engine/pytorch/csrc/extensions/pybind.cpp
  m.def("quantize", transformer_engine::pytorch::quantize, py::arg("tensor"), py::arg("quantizer"),
        py::arg("output") = py::none(), py::arg("noop") = py::none());

transformer_engine::pytorch::quantize @ transformer_engine/pytorch/csrc/extensions/cast.cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<at::Tensor> noop_flag) {

    data conversion here
    // Perform quantization
    quantize_impl(input_cpp, quantizer, quantizer_cpp, output_cpp, noop_flag_cpp);

    quantize_impl is not just a function at the beginning of cast.cpp

        nvte_quantize_v2(input.data(), output.data(), **quant_config**, at::cuda::getCurrentCUDAStream());

        transformer_engine/common/util/cast.cu
        void nvte_quantize_v2(const NVTETensor input, NVTETensor output, const NVTEQuantizationConfig quant_config, cudaStream_t stream) 

            detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>(
            i   input, grad, output, dbias, workspace, quant_config, stream);

            util/cast_kernels.cuh:1218: void quantize_helper(const NVTETensor input, const NVTETensor grad, NVTETensor output,
                    NVTETensor dbias, NVTETensor workspace,
                    const NVTEQuantizationConfig quant_config, cudaStream_t stream) 

                    case NVTE_MXFP8_1D_SCALING: {
                    mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
                        *input_tensor, activation_input_tensor, &noop_tensor, output_tensor, dbias_tensor,
                        workspace_tensor, stream);

                    template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
                    float (*OP)(float, const ParamOP &)>
                    void mxfp8_quantize(const Tensor &input, const Tensor *act_input,
                                        const Tensor *noop,  // TODO (ksivamani)
                                        Tensor *output, Tensor *dbias, Tensor *workspace, cudaStream_t stream) {

Questions: who set NVTE_MXFP8_1D_SCALING?
    switch (output_tensor->scaling_mode) {

transformer_engine/common/include/transformer_engine/transformer_engine.h
    NVTEScalingMode type


need to come back:
1. void CheckScaleTensorShape(const Tensor &t, const std::string &name) {
    this is alignment?

ignore for now
common/include/transformer_engine/activation.h:23: *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,


106:  m.def("bgrad_quantize", transformer_engine::pytorch::bgrad_quantize,
        "Compute bias gradient and quantize", py::arg("input"), py::arg("quantizer"));

generic_gemm
    * in python general_gemm
        forward path, layout is not provided, default to TN, meaning transa True, Transb False.

    * transformer_engine/pytorch/csrc/extensions/pybind.cpp
        m.def("generic_gemm", transformer_engine::pytorch::gemm, "Compute GEMM (matrix-matrix multiply)",
                py::arg("A"), py::arg("transA"), py::arg("B"), py::arg("transB"), py::arg("D"),
                py::arg("quantizer"), py::arg("output_dtype"), py::arg("bias"), py::arg("bias_type"),
                py::arg("gelu"), py::arg("gelu_in"), py::arg("grad"), py::arg("workspace"),
                py::arg("workspace_size"), py::arg("accumulate"), py::arg("use_split_accumulator"),
                py::arg("comm_overlap") = nullptr, py::arg("comm_type") = std::nullopt,
                py::arg("extra_output") = std::nullopt, py::arg("bulk_overlap") = false);

    * where is transformer_engine::pytorch::gemm
        transformer_engine/pytorch/csrc/extensions/gemm.cpp

        std::vector<py::object> gemm(   py::handle A, bool transa, py::handle B, bool transb, py::object D,
                                        py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                                        DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                                        at::Tensor workspace, size_t workspaceSize, bool accumulate,
                                        bool use_split_accumulator, CommOverlapCore* comm_overlap,
                                        std::optional<CommOverlapType> comm_type, MaybeTensor extra_output,
                                        bool bulk_overlap) {



# Annotated: mxfp8_quantize (TransformerEngine)

File: `TransformerEngine/transformer_engine/common/util/cast_kernels.cuh`

Below is a commented, read-only-style excerpt of only the `mxfp8_quantize` function, with inline notes explaining each line or block.

```cpp
// Template params control fused behaviors and activation op selection.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)> 
void mxfp8_quantize(const Tensor &input, const Tensor *act_input,
                    const Tensor *noop,  // forwarded auxiliary tensor (often identity)
                    Tensor *output, Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  // Which directions we will emit quantized data for.
  bool use_rowwise_scaling = output->has_data();
  bool use_colwise_scaling = output->has_columnwise_data();

  // Ensure CUDA driver context is set up for this stream.
  checkCuDriverContext(stream);

  // Input must carry rowwise data; output must be FP8 for MXFP8 path.
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");

  // If we will write rowwise or colwise scales, their destination buffers must exist.
  if (use_rowwise_scaling) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  }
  if (use_colwise_scaling) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated");
  }

  // Validate the auxiliary "noop" tensor contract.
  CheckNoopTensor(*noop, "cast_noop");

  // MXFP8 uses 32-sized blocks along the scaling dimension(s); 1 means not used.
  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? 32 : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? 32 : 1;

  // Flattened problem size and tiling configuration for MXFP8 kernels.
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, MXFP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, MXFP8_CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, MXFP8_CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, MXFP8_CHUNKS_PER_BLOCK_X);

  // Strides into the scale tensors (2D layout) where per-block scales will be written.
  const size_t scale_stride_rowwise = use_rowwise_scaling ? output->scale_inv.shape[1] : 1;
  const size_t scale_stride_colwise =
      use_colwise_scaling ? output->columnwise_scale_inv.shape[1] : 1;

  // Pointers to scale tensors. Null if not used in that direction.
  e8m0_t *const scales_rowwise_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      use_colwise_scaling ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  // dbias (columnwise sum) workspace geometry: one row per block in Y, one column per data column.
  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  if constexpr (IS_DBIAS) {
    // dbias must match input dtype, and is a 1D vector of length 'cols'.
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    // First sizing pass: if workspace isn't allocated yet, publish required shape/dtype and return.
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }

  // Optional dbias accumulation buffer; amax accumulator for FP8 stats.
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);

  // Launch configuration for MXFP8 2D kernel.
  const dim3 block(MXFP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  // Instantiate kernel with compile-time scaling tile sizes and I/O types.
  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
                  output->dtype(), OType,

                  // TMA tensor maps for input, optional act_input, and outputs.
                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_act_input{};
                  alignas(64) CUtensorMap tensor_map_output_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_colwise{};

                  // Describe input to TMA (extents, tile shape, stride, element bits).
                  create_2D_tensor_map(tensor_map_input, input.data, rows, cols, MXFP8_SHMEM_DIM_Y,
                                       MXFP8_SHMEM_DIM_X, cols, 0, typeToNumBits(input.dtype()));

                  // Optional second source for fused derivative-of-activation path.
                  if constexpr (IS_DACT) {
                    create_2D_tensor_map(tensor_map_act_input, act_input->data, rows, cols,
                                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X, cols, 0,
                                         typeToNumBits(input.dtype()));
                  }

                  // Outputs for rowwise and/or colwise quantized tensors.
                  if (use_rowwise_scaling) {
                    create_2D_tensor_map(tensor_map_output_rowwise, output->data, rows, cols,
                                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X, cols, 0,
                                         typeToNumBits(output->dtype()));
                  }

                  if (use_colwise_scaling) {
                    create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data, rows,
                                         cols, MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X, cols, 0,
                                         typeToNumBits(output->dtype()));
                  }

                  // Main Hopper/TMA kernel: computes scales and FP8 outputs (and optional dbias).
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType,
                                       SCALE_DIM_Y, SCALE_DIM_X><<<grid, block, 0, stream>>>(
                      tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                      tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr,
                      reinterpret_cast<const float *>(noop->data.dptr), workspace_ptr, amax_ptr,
                      rows, cols, scale_stride_rowwise, scale_stride_colwise);

                  // If enabled, reduce workspace [dbias_rows, cols] to dbias[cols].
                  if constexpr (IS_DBIAS) {
                    reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
                  }));  // type-switch macros close
          ));
}
```
InTypeAB,     , OutType,    , ComputeType, ScaleType,     DScaleType, InTypeC
__nv_fp4_e2m1, __nv_fp4_e2m1, float,       __nv_fp8_e4m3, float,      __nv_bfloat16


InTypeAB = __nv_fp4_e2m1
OutType  = __nv_fp4_e2m1
ComputeType = float
ScaleType = __nv_fp8_e4m3
DScaleType = float
InTypeC = __nv_bfloat16

pytorch/csrc/extensions/gemm.cpp
    std::vector<py::object> gemm( 
        swizzle_scaling_factors [ ]
        Tensor allocator [ ]
        common/gemm/cublaslt_gemm.cu
        void nvte_cublas_gemm
            convertNVTETensorCheck [ ]
            convertNVTETensor [ ]
            void cublas_gemm (common interface) [x] sort of
                CanonicalizeGemmInput (support nvfp4) [x]







refactor TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH
TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH

TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY, add FP4? 

question: how it is aligned nicely

CheckScaleTensorShape




InTypeAB = __nv_fp4_e2m1,
OutType = __nv_fp4_e2m1, 
ComputeType = float, 
ScaleType = __nv_fp8_e4m3, 
DScaleType = float, 
InTypeC = __nv_bfloat16

__nv_fp8_e4m3, __nv_fp8_e4m3, float, __nv_fp8_e8m0, __nv_fp8_e8m0, __nv_bfloat16>


mxfp8
name=NVIDIA B200 cc=10.0
cuBLAS version=120901
heuristic_returned=1  first_ws=0
compute=68 scale=0 transA=1 transB=0 epilogue=4 ptrMode=0
A_scale_mode=2 ptr=0x709ee98c4800 | B_scale_mode=2 ptr=0x709ee98c4a00 | C_scale_mode=0 ptr=(nil) | D_scale_mode=0 ptr=(nil)
A: type=28 order=0 rows=64 cols=64 ld=64 batch=1 stride=0
B: type=28 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
C: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
D: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0



nvfp4
name=NVIDIA B200 cc=10.0
cuBLAS version=120901
heuristic_returned=0  first_ws=0
compute=68 scale=0 transA=1 transB=0 epilogue=4 ptrMode=0
A_scale_mode=1 ptr=0x70db4f8b3800 | B_scale_mode=1 ptr=0x70db4f8b3a00 | C_scale_mode=0 ptr=(nil) | D_scale_mode=0 ptr=(nil)
A: type=33 order=0 rows=64 cols=64 ld=64 batch=1 stride=0
B: type=33 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
C: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
D: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0




at pytorch level, 


Quantizer (builder class, base)
    bool rowwise_usage
    bool columnwise_usage
    bool internal : Whether to instantiates tensor for purely internal usage (open, hardcoded to false in base clase, MXFP8Quantizer does not expose it as well)

    quantize function is not an abstract method, it calls autograd _QuantizeFunc which wrap tex.quantize. 
        tex.quantize requires input tensor B/FP16/32 and the quantizer

    during quantizer initialization, it sets the target dtype by just dtype
    pretty straightforward quantizer, rowwise, columnwise are default to True. 
    internal = True when Linear._get_quantizers()/_get_debug_quantizers() is called

    

how Quantizers are instantiated?
* RecipeState (come back later)
* make_quantizers method which based on num_quantizers

when are they instantiated? prepare_forward of te.linear, right before forward
each te.linear has two sets of quantizers, one for forward, one for backward
"scaling_fwd" and "scaling_bwd"
but only 3 are being used for mxfp8, because we only quantize input, weight, and grad_output

only 3 quantizers for mxfp8 and nvfp4, why? tbd
                input_quantizer,      self.quantizers["scaling_fwd"][0]
                weight_quantizer,     self.quantizers["scaling_fwd"][1]  
                output_quantizer,      # None
                grad_input_quantizer,  # None
                grad_weight_quantizer, # None
                grad_output_quantizer, self.quantizers["scaling_bwd"][0]

mapping of idx
  py::enum_<transformer_engine::pytorch::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_OUTPUT)
      .value("GEMM3_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM3_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_OUTPUT);

  py::enum_<transformer_engine::pytorch::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT2)
      .value("GRAD_OUTPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_INPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT3);


tex.quantize(tensor, quantizer) the last two are implicitly None      
    @ TransformerEngine/transformer_engine/pytorch/csrc/extensions/cast.cpp
    1. create cpp MXFP8Quantizer from quantizer handle wrapped in a pointer
    1. input tensor is ensured to be contiguous
    1. input_cpp, wrap torch input tensor with TensorWrapper(NVTETensor/Tensor internally)
        L106: transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor)
    1. output_cpp, also a TensorWrapper. Because output is None,
        MXFP8Quantizer->create_tensor(shape, dtype (fake_dtype because it is not used), optional rowwise data)
        here it creates
        at::Tensor rowwise_data1, columnwise_data, rowwise_scale_inv, columnwise_scale_inv; 
        opts.dtype(torch::kUInt8) byte size datatype
        create data, rowwise_scale_inv based on rowwise_usage 
        create columnwise_data, columnwise_scale_inv based on columnwise_usage
        std::pair<TensorWrapper, py::object> MXFP8Quantizer::create_tensor(
            L469 TransformerEngine/transformer_engine/pytorch/csrc/quantizer.cpp
    1. output_py is returned which is a pybind object

