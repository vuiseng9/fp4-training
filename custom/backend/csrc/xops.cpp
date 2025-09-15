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
  m.def("cublaslt_mm_fp32_or_bf16(int gemm_id, Tensor A, Tensor B, Tensor? b=None) -> Tensor"); // A, B row-major for all 3 modes. 0=fwd_gemm, 1=input grad gemm, 2=weight grad gemm
  m.def("cublaslt_mm_mxfp8(int gemm_id, ScalarType DoutType, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB, Tensor? b=None) -> (Tensor Dt, Tensor scaleDout)"); // A, B row-major for all 3 modes. 0=fwd_gemm, 1=input grad gemm, 2=weight grad gemm
  m.def("addmm_cuda(Tensor a, Tensor b, Tensor? c=None) -> Tensor");
  m.def("cublaslt_mxfp8_matmul_bias_epilogue(Tensor row_major_W, Tensor row_major_W_scale_swizzled, Tensor col_major_Xt, Tensor row_major_X_scale_swizzled, Tensor? b=None) -> Tensor"); // expected W[OC, IC] row_major, X.T[IC, N] col_major, return Y[N, OC] row major
  m.def("cublaslt_matmul_bias_epilogue(Tensor row_major_W, Tensor col_major_Xt, Tensor? b=None) -> Tensor"); // expected W[OC, IC] row_major, X.T[IC, N] col_major, return Y[N, OC] row major
  m.def("cublaslt_matmul_xbias(Tensor any_major_A, Tensor any_major_B) -> Tensor"); // internally forcing row-major layout through .contiguous()
}