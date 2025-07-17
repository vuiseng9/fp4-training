



https://github.com/NVIDIA/TransformerEngine


docker run -d --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3


install-torch 126
install-cuda-toolkit-conda 12.6
conda install -c nvidia cudnn=9.10.2
cp -r /home/shadeform/miniforge3/envs/sf-250714-te/lib/python3.12/site-packages/nvidia/nvtx/include/nvtx3 /home/shadeform/miniforge3/envs/sf-250714-te/targets/x86_64-linux/include/.

MAX_JOBS=10 pip install --no-build-isolation transformer_engine[pytorch]

https://anaconda.org/nvidia/cudnn
copy nvtx3 folder to cuda.h folder
compile with more thread

Good Documentation:
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html



GTC March 2025: https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/
June 5 2025: 
https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/
Jun 24, 2025
https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/



AMD FP8 blog (March 13, 2025)
https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html