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

### Build TransformerEngine from source
1. Using nvidia:cuda container (challenge, need corresponding build of PyTorch and HW)
    ```bash
    apt install -y cmake htop
    pip install pybind11

    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    git submodule update --init --recursive

    export NVTE_FRAMEWORK=pytorch         # Optionally set framework
    MAX_JOBS=$(nproc) pip3 install -v --no-build-isolation .   # Build and inst
    # it takes a while, pls be patient, even it seems quiet in htop
    ```
1. Using Conda (challenge: hard to get bleeding-edge features)
    ```bash
    install-torch 126
    install-cuda-toolkit-conda 12.6
    conda install -c nvidia cudnn=9.10.2
    cp -r /home/shadeform/miniforge3/envs/sf-250714-te/lib/python3.12/site-packages/nvidia/nvtx/include/nvtx3 /home/shadeform/miniforge3/envs/sf-250714-te/targets/x86_64-linux/include/.

    MAX_JOBS=10 pip install --no-build-isolation transformer_engine[pytorch]

    https://anaconda.org/nvidia/cudnn
    copy nvtx3 folder to cuda.h folder
    compile with more thread
    ```
