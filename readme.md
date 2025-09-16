### Quantized Training

Scope of the repo:
1. Minimal examples and notes illustrating model training across floating point and microscaling precision formats.
2. Integrates Experimental custom kernel for quantized training, mostly target functionality first, performance second.

As of September 2025, PyTorch natively supports training down to 16-bit.
Best Options to train in FP8:
* NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
* PyTorch [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8#training-benchmarks)

Most examples packaged here are mostly Transformer Engine first.

> Use docker image from nvidia `nvcr.io/nvidia/pytorch:25.06-py3` where pytorch, Transformer Engine etc are coherent. No installation required to script in this repo, just remember to set PYTHONPATH=/path/to/quantized-training. For AMD, use thier megatron-lm docker image `rocm/megatron-lm:v25.6_py312`, megatron-lm depends on Transformer Engine,

Nvidia Transformer Engine Notes:
1. While RTX 50 series and RTX PRO 6000 PRO GPUs have compute capability of 12.0 where HW comes with FP8 and lower, CUDA libs such as cuBLASLt also can run MXFP8/NVFP4, Transformer Engine limits its build to B200. Therefore, TE mxfp8 recipe (`MXFP8BlockScaling`) will not work on these devices.
2. FP8 block scaling (`Float8BlockScaling`) recipe is only meant for Hopper, it won't work on other non-Hopper architectures, even Blackwell. Believe that it is limited by software implementation rather than hardware.

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
1. [2025/08/29, Nvidia's on Fine-Tuning gpt-oss with MXFP4/NVFP4 QAT (weight and/or activation)](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/) via [Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss)
1. [2025/08/25, Nvidia's on NVFP4/MXFP4 Training of Mamba-Transformer](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), no code shared yet, most likely via through research kernel like Quartet/Qutlass.
1. [2025/06/05, Nvidia's on FP8 Training via Transformer Engine](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
1. [2025/06/24, Nvidia's on NVFP4 Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
1. [2025/03/13, AMD's blog on FP8 Training](https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html)

Tailored-made kernels:
1. [2025/08/19, Cursor's on 1.5x Faster MoE Training with Custom MXFP8 Kernels](https://cursor.com/en/blog/kernels#building-the-fastest-mxfp8-quantization-kernel-ever)
2. 2025/08/28, Mojo's Matrix Multiplication on Blackwell, [Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
3. 2025/09/05, Mojo's Matrix Multiplication on Blackwell, [Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul)
4. 2025/09/12, Mojo's Matrix Multiplication on Blackwell, [Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
5. [2023/03/23, Mojo AI's Compute Fragmentation: What Matrix Multiplication Teaches Us](https://www.modular.com/blog/ais-compute-fragmentation-what-matrix-multiplication-teaches-us)

Notes:
1. [Aug State](./notes/250801_State_of_Narrow_Precision.md)
1. See [SOTA Research](./notes/99_sota_notes.md)
1. [Build Transformer Engine from source](https://github.com/vuiseng9/vs-colab/blob/main/cheatsheet/tx-engine.md)
