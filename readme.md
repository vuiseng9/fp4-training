### Quantized Training

Minimal examples and notes illustrating model training across floating point and microscaling precision formats.
As of July 2025, PyTorch natively supports training down to 16-bit; NVIDIA’s [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) is required for FP8 and below.

### Fastest Setup
```bash
# Transformer Engine included.
docker run -d --gpus all -it --rm nvcr.io/nvidia/pytorch:25.06-py3
```

Useful References:
1. [Tranformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
1. [Talk in GTC March 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
1. [2025/06/05, Nvidia's blog on FP8 Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
1. [2025/06/24, Nvidia's blog on NVFP4 Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
1. [2025/03/13, AMD's blog on FP8 Training](https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html)

TODO SOTA Research:
