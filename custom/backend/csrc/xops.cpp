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
  m.def("cublaslt_linear(Tensor W, Tensor X, Tensor? b=None) -> Tensor"); // expected W[OC, IC], X[N, IC], return [N, OC] == A[x, k], B[y, k], c[x]  B @ A.T = O[y, x]
}