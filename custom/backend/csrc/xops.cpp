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
}