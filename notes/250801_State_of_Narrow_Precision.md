
The State of Training in Narrow Precision (FP8 & below)

1. If a company or a user to train a model in narrow precision, what is the path of least resistance?
    * *FP8 is maturing*
        * Use Nvidia PyTorch Container (tag: `25.06-py3`). It packs `transformer_engine` v2.4. It provides the following:
            1. Per tensor scaling: a) `DelayedScaling`, b) `Float8CurrentScaling`. These works on any arch above Ada Lovelace.
			2. `Float8BlockScaling` - `TE` limits this to Hopper only. 
                <span style="color: red;"> In blogs and GTC presentation, there were different block size, are they officially supported? it may be is in cublas but not exposed in TE? text</span>.
			3. `MXFP8BlockScaling`: Tested working on B200, but not on RTX50 series. It appears to me that `TE` does not expose the functionality. HW arch certainly supports, cuBLAS and cutlass have examples on this precision.
            4. In short, Default to `MXFP8BlockScaling` on B200/GB200. On Hopper, `DelayScaling` → `CurrentScaling` → `Float8BlockScaling`; move down the list only if convergence is unstable.
    * If we choose AMD, what can we do?
        1. Use ROCm Megatron-LM which packages respective TE, MI300. Currently only supports `DelayScaling`. No `Float8CurrentScaling`, `Float8BlockScaling`, `MXFP8BlockScaling` (not sure, i tested on MI300, fundamentally not supporting MX format). See [ROCm 6.4.2 released Mid July, Megatron-LM, Transformer Engine](
        https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html?model=pyt_megatron_lm_train_llama-3.1-8b)
        `docker pull rocm/megatron-lm:v25.6_py312`

2. Software Enabling for MXFP8/6/4
    * Nvidia CUBLAS [12.9 blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9)
        - Channel- and block-scaled FP8 matmuls on NVIDIA Hopper
	    - outer-vector scaling - https://docs.nvidia.com/cuda/cublas/#outer-vector-scaling-for-fp8-data-types (1x128, 128x128)
	    - refer to the cuBLASLt Library API examples. [nvfp4](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul), [mxfp8](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtMxfp8Matmul)
    * Nvidia CUTLASS
        * https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm
		* https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm
    * AMD: FP6/FP4 are supported at [Composable Kernel level](
https://rocm.docs.amd.com/projects/composable_kernel/en/latest/reference/Composable_Kernel_supported_scalar_types.html)
        https://github.com/ROCm/composable_kernel/blob/develop/example/67_gemm_microscaling/gemm_mx_fp6.cpp 
    * TransformerEngine Deep Dive
        * transformer_engine/common/util/cast_kernels.cuh, see mxfp8_quantize
        * /home/vs9/temp/TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu, line 588 cublasLtMatmul(, call cuBLASLt api




Inference Front

Questions
any blackwell training perf data?
* Nope, not yet

any training perf data with Hopper?
GTC2503, H100, 8,70,405B of llama3.1
1.32, 1.44, 1.51X over BF16 (FP8 per tensor)
GTC2503, just projection on other granularity based on DeepSeek



https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html#low-precision-fp8-multi-node-scaling-performance
https://hub.docker.com/r/rocm/7.0-preview
LLama2-70B LoRA finetuning (llama2 - is it fp8)




on inference front, what are model trained in FP8? then quantized to FP4, results?

DeepSeek-R1 (accuracy, no performance?)
https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference


https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/
SVDQUANT: ABSORBING OUTLIERS BY LOW-RANK COMPONENTS FOR 4-BIT DIFFUSION MODELS
first look, performance doesnt look great

MXFP4/MXFP8 (Quarks) - how do they fare in accuracy?
https://huggingface.co/collections/amd/quark-quantized-mxfp4-models-68068f8c965d9267a996616d
https://huggingface.co/collections/amd/quark-quantized-ocp-fp8-models-66db7936d18fcbaf95d4405c

Feb 24, 2025, inference rocm fp8 vllm
per token activation, per channel activation
https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html


DeepSeek is an engineering marvel






