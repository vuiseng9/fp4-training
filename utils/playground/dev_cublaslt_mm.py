import torch
from torch import nn
import backend.xops
from custom import CublasltLinear

op = torch.ops.xops


def cuda_rand_tensor(shape, dtype=torch.bfloat16):
    return torch.rand(*shape).cuda().to(dtype)

def max_error_element(x, ref):
    return (x-ref).abs().max()

def assert_allclose(x, ref, atol=1e-5):
    assert torch.allclose(x, ref, atol=atol), f"max error {max_error_element(x, ref)} > {atol}"

B = 4
L = 16
IC = 128
OC = 64


# X = torch.rand(B*L, IC).cuda().to(torch.float32)
# W = torch.rand(OC, IC).cuda().to(torch.float32)

X = cuda_rand_tensor((B*L, IC))
W = cuda_rand_tensor((OC, IC))
gradY = cuda_rand_tensor((B*L, OC))
b = cuda_rand_tensor((OC,))

# Gemm 1, fwd -----------------------------------------------------
Y_test = op.cublaslt_mm_fp32_or_bf16(0, W, X, b)
Y_ref = torch.addmm(b, X, W.T) 
print(f"max element error: {max_error_element(Y_test, Y_ref)}")
assert_allclose(Y_test, Y_ref)

# no bias
b = None
Y_test = op.cublaslt_mm_fp32_or_bf16(0, W, X, b)
Y_ref = torch.mm(X, W.T)
print(f"max element error: {max_error_element(Y_test, Y_ref)}")
assert_allclose(Y_test, Y_ref)

# Gemm 2, grad X -----------------------------------------------------
gradX_test = op.cublaslt_mm_fp32_or_bf16(1, W, gradY)
gradX_ref = torch.mm(gradY,   W)
print(f"max element error: {max_error_element(gradX_test, gradX_ref)}")
assert_allclose(gradX_test, gradX_ref)

# Gemm 3, grad W -----------------------------------------------------
gradW_test = op.cublaslt_mm_fp32_or_bf16(2, X, gradY)
gradW_ref = torch.mm(gradY.T, X)
print(f"max element error: {max_error_element(gradW_test, gradW_ref)}")
assert_allclose(gradW_test, gradW_ref)


# CustomMatMul
# CublasltLinearFunc
dtype = torch.float32
useB = False
torch_linear = nn.Linear(IC, OC, bias=useB).to(device="cuda", dtype=dtype)
custom_linear = CublasltLinear(IC, OC, bias=useB).to(device="cuda", dtype=dtype)
custom_linear.load_state_dict(torch_linear.state_dict())


# x = torch.randn(B, L, IC).to(device="cuda", dtype=dtype)
x_test = cuda_rand_tensor((B*L, IC), dtype=dtype)
x_ref = x_test.clone() # we need another copy, otherwise, backward gonna accumulate on same tensor
x_test.requires_grad = True
x_ref.requires_grad = True

y_test = custom_linear(x_test)
y_ref = torch_linear(x_ref)

print(f"max element error: {max_error_element(y_test, y_ref)}")
assert_allclose(y_test, y_ref)

y_test.sum().backward()
y_ref.sum().backward()

print(f"max element error: {max_error_element(torch_linear.weight.grad, custom_linear.weight.grad)}")
assert_allclose(torch_linear.weight.grad, custom_linear.weight.grad)

print(f"max element error: {max_error_element(x_test.grad, x_ref.grad)}")
assert_allclose(x_test.grad, x_ref.grad)
print("joto")

