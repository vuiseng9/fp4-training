from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

setup(
      # name="custom_ops", move to pyproject.toml
      # version="0.0.1", move to pyproject.toml
      ext_modules=[
          cpp_extension.CUDAExtension(
            name="backend.xops", 
            sources=[
                "backend/csrc/xops.cpp", 
                "backend/csrc/aten_mm.cpp"
                ]
            )],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)