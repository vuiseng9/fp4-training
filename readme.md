### Quantized Training

Minimal examples and notes illustrating model training across floating point and microscaling precision formats.
As of July 2025, PyTorch natively supports training down to 16-bit; 

Options to train in FP8:
* NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
* PyTorch [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8#training-benchmarks)

Most examples packaged here are mostly Transformer Engine first.

> no installation required, just remember to set PYTHONPATH=/path/to/quantized-training

### Fastest Setup
```bash
# Transformer Engine included.
docker run -d --gpus all -it --rm nvcr.io/nvidia/pytorch:25.06-py3

# this version support transformer engine v2.4+, FP8 Block scaling only on Hopper, MXFP8 only on Blackwell, RTX50 series not supported in TE but HW does have them, use CUTLASS
```
AMD
```bash
# IMG_NAME=rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0 # no TE

# docker run -d -it \
#     --cap-add=SYS_PTRACE \
#     --security-opt seccomp=unconfined \
#     --device=/dev/kfd \
#     --device=/dev/dri \
#     --group-add video \
#     --ipc=host \
#     --shm-size 8G \
#     $IMG_NAME

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
1. [2025/06/05, Nvidia's blog on FP8 Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
1. [2025/06/24, Nvidia's blog on NVFP4 Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
1. [2025/03/13, AMD's blog on FP8 Training](https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html)

TODO SOTA Research:

[Pre-training with float8 using torchtitan and torchao](https://docs.pytorch.org/ao/0.12/pretraining.html#pre-training-with-torchtitan)
Quick note: Torchao supports float8 rowwise and tensorwise.

### Build TransformerEngine from source
1. Using nvidia:cuda container (challenge, need corresponding build of PyTorch and HW, TE and Torch CUDA must match)
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
