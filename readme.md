### Quantized Training

Scope of the repo:
1. Minimal examples and notes illustrating model training across floating point and microscaling precision formats.
2. Integrates Experimental custom kernel for quantized training.

As of September 2025, PyTorch natively supports training down to 16-bit.
Best Options to train in FP8:
* NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
* PyTorch [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8#training-benchmarks)

Most examples packaged here are mostly Transformer Engine first.

> Use docker image from nvidia `nvcr.io/nvidia/pytorch:25.06-py3` which pytorch, Transformer Engine etc. no installation required to script in this repo, just remember to set PYTHONPATH=/path/to/quantized-training. For AMD, use thier megatron-lm docker image `rocm/megatron-lm:v25.6_py312`, megatron-lm depends on Transformer Engine,

> Important: 
> 1. while RTX 50 series GPUs have compute capability of 12.0, supporting down to FP4, it is not exposed by Transformer Engine. Therefore, TE mxfp8 recipe will not work on RTX 50 series.
> 2. FP8 block scaling recipe of TE is only meant for Hopper, it won't work of Blackwell. Believe that it is limited by software implementation rather than hardware.

[AMD docker image](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html?model=primus_pyt_megatron_lm_train_llama-3.3-70b)
```bash
IMG_NAME=rocm/megatron-lm:v25.6_py312
docker run -d -it \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 128G \
    --name megatron_training_env \
    $IMG_NAME
    # -v $HOME:$HOME \
```


Useful References:
1. [Tranformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
1. [Talk in GTC March 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
1. [Pre-training with float8 using torchtitan and torchao](https://docs.pytorch.org/ao/0.12/pretraining.html#pre-training-with-torchtitan)

Blogs:
1. [2025/08/29, Nvidia's on Fine-Tuning gpt-oss with MXFP4/NVFP4 QAT (weight and/or activation)](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/)
1. [2025/08/25, Nvidia's on NVFP4/MXFP4 Training of Mamba-Transformer](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
1. [2025/08/19, Cursor's on 1.5x Faster MoE Training with Custom MXFP8 Kernels](https://cursor.com/en/blog/kernels#building-the-fastest-mxfp8-quantization-kernel-ever)
1. [2025/06/05, Nvidia's on FP8 Training via Transformer Engine](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
1. [2025/06/24, Nvidia's on NVFP4 Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
1. [2025/03/13, AMD's blog on FP8 Training](https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html)

Notes:
1. [Aug State](./notes/250801_State_of_Narrow_Precision.md)
1. See [SOTA Research](./notes/99_sota_notes.md)
1. [Build Transformer Engine from source](https://github.com/vuiseng9/vs-colab/blob/main/cheatsheet/tx-engine.md)
