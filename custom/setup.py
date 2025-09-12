from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

# Debug flags for C++
cpp_debug_flags = [
    '-g',           # Generate debug symbols
    '-O0',          # No optimization
    '-DDEBUG',      # Define DEBUG macro
    '-fno-inline',  # Disable inlining for better stack traces
    '-fno-omit-frame-pointer',  # Keep frame pointer for debugging
]

# Debug flags for CUDA
cuda_debug_flags = [
    '-g',           # Device code debug info
    '-G',           # Host code debug info
    '-O0',          # No optimization
    '-DDEBUG',      # Define DEBUG macro
    '--generate-line-info',  # Line number info for profilers
    '-Xcompiler', '-fno-inline',  # Pass to host compiler
]

# Check if we're in debug mode
# DEBUG_MODE = os.environ.get('DEBUG_BUILD', '0') == '1'

setup(
    # name="custom_ops", move to pyproject.toml
    # version="0.0.1", move to pyproject.toml
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="backend.xops", 
            sources=[
                "backend/csrc/xops.cpp", 
                "backend/csrc/aten_mm.cpp",
                "backend/csrc/cublaslt_mm.cu",
                "backend/csrc/cublaslt_mm_fp32bf16.cu"
            ],
            extra_compile_args={
                'cxx': cpp_debug_flags,
                'nvcc': cuda_debug_flags,
            },
            # Important: PyTorch's default flags might override, so we force debug, TODO use a different flag
            define_macros=[('DEBUG', None)],
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    # 'build_ext': BuildExtension.with_options(
    #     no_python_abi_suffix=True,
    #     use_ninja=False  # Ninja can sometimes interfere with debug builds
    # )
)