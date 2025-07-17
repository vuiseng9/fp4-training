install-torch 126
install-cuda-toolkit-conda 12.6
conda install -c nvidia cudnn=9.10.2
cp -r /home/shadeform/miniforge3/envs/sf-250714-te/lib/python3.12/site-packages/nvidia/nvtx/include/nvtx3 /home/shadeform/miniforge3/envs/sf-250714-te/targets/x86_64-linux/include/.

MAX_JOBS=10 pip install --no-build-isolation transformer_engine[pytorch]

https://anaconda.org/nvidia/cudnn
copy nvtx3 folder to cuda.h folder
compile with more thread