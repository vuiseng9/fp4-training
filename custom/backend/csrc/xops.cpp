#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" PyObject* PyInit_xops(void) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "xops", nullptr, -1, nullptr,
  };
  return PyModule_Create(&def);
}

TORCH_LIBRARY(xops, m) {
  m.def("addmm_cuda(Tensor a, Tensor b, Tensor? c=None) -> Tensor");
  m.def("cublaslt_mm_fp32_or_bf16(int gemm_id, Tensor A, Tensor B, Tensor? b=None) -> Tensor"); // A, B row-major for all 3 modes. 0=fwd_gemm, 1=input grad gemm, 2=weight grad gemm
  m.def("cublaslt_mm_mxfp8(int gemm_id, ScalarType DoutType, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB, Tensor? b=None) -> (Tensor Dt, Tensor scaleDout)"); // A, B row-major for all 3 modes. 0=fwd_gemm, 1=input grad gemm, 2=weight grad gemm
  m.def("cublaslt_mm_nvfp4(int gemm_id, ScalarType DoutType, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB, Tensor? b=None, float? alpha=None) -> (Tensor Dt, Tensor scaleDout)"); // A, B row-major, only TN layout is supported
}