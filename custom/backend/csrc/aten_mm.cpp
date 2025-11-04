#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>


namespace xops {

at::Tensor addmm_cuda(const at::Tensor& a, const at::Tensor& b, c10::optional<at::Tensor> c) {

  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b tensors must be CUDA");
  TORCH_CHECK(a.device() == b.device(), "a, and b must be on the same CUDA device");

  const auto dtype = a.scalar_type();
  TORCH_CHECK((dtype == at::kFloat || dtype == at::kBFloat16), "dtype must be float32 or bfloat16; but is ", dtype);
  TORCH_CHECK(a.scalar_type() == dtype && b.scalar_type() == dtype, "a and b must share the same dtype");

  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
  const auto m = a.size(0);
  const auto k = a.size(1);
  TORCH_CHECK(b.size(0) == k, "matmul inner dims must match: a(:,", k, ") vs b(", b.size(0), ",:)");
  const auto n = b.size(1);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  
  if (c) {
    TORCH_CHECK(c->is_cuda(), "c must be CUDA if provided");
    TORCH_CHECK(c->device() == a.device(), "c must be on same device as a/b");
    TORCH_CHECK(c->scalar_type() == dtype, "c must match dtype of a/b");
    TORCH_CHECK(c->dim() == 1 && c->size(0) == n,
                "c must be 1D of length ", n, ", got ", c->sizes());
    
    at::Tensor c_2d = c->view({1, n}).expand({m, n}).contiguous();

    return at::addmm(c_2d, a_contig, b_contig);
  
  } else {
  
    return at::mm(a_contig, b_contig);
  
  }
}

}

// Registers CUDA implementations for addmm_cuda
TORCH_LIBRARY_IMPL(xops, CUDA, m) {
  m.impl("addmm_cuda", xops::addmm_cuda);
}