import torch
from torch import nn
import backend.xops
from custom import CublasltMxfp8Linear

op = torch.ops.xops

from custom.backend.triton.mxfp8 import quantize_rowwise, quantize_colwise, mxfp8_matmul_NT, pad_to_multiple

def cuda_rand_tensor(shape, dtype):
    return torch.rand(shape, dtype=dtype, device="cuda")
    
def max_error_element(x, ref):
    # Absolute error
    abs_err = (x - ref).abs()
    absmax_val = abs_err.max()
    abs_flat = torch.argmax(abs_err)
    abs_y, abx_x = torch.unravel_index(abs_flat, abs_err.shape)
    abs_coords = (abs_y.item(), abx_x.item())

    # Relative error
    rel_err = (x / (ref + 1e-6) - 1).abs()
    relmax_val = rel_err.max()
    rel_flat = torch.argmax(rel_err)
    rel_y, rel_x = torch.unravel_index(rel_flat, rel_err.shape)
    rel_coords = (rel_y.item(), rel_x.item())

    return f"absmax: {absmax_val.item()} at {abs_coords}, relmax: {relmax_val.item()} at {rel_coords}"

input_dtype = torch.bfloat16
out_dtype = torch.bfloat16  

tolerance = 1e-1

# M, N, K = 8192, 4096, 1024
# W = torch.rand((M, K), dtype=input_dtype, device="cuda")
# X = torch.rand((N, K), dtype=input_dtype, device="cuda")
# W     = cuda_rand_tensor((M, K), dtype=input_dtype)
# X     = cuda_rand_tensor((N, K), dtype=input_dtype)
B = 4
L = 64
B = 8
L = 15
IC = 128
OC = 512
W     = cuda_rand_tensor(( OC, IC), dtype=input_dtype)
X     = cuda_rand_tensor((B*L, IC), dtype=input_dtype)

outM = W.shape[0]
outN = X.shape[0]
print(f"original shape: W {W.shape}, X {X.shape}")
W = pad_to_multiple(W, ty=128, tx=128)
X = pad_to_multiple(X, ty=256, tx=128)

# Gemm 1, fwd -----------------------------------------------------
b = None
Wq, scaleW = quantize_rowwise(W)
Xq, scaleX = quantize_rowwise(X)
Yt_test = mxfp8_matmul_NT(Wq, scaleW, Xq, scaleX, torch.bfloat16)
print(f"after padding: W {W.shape}, X {X.shape}, Y {Yt_test.shape}")
Yt_test = Yt_test[:outM, :outN]

Yt_ref = torch.matmul(W, X.T)
Yt_ref = Yt_ref[:outM, :outN]

print(f"max element error: {max_error_element(Yt_test, Yt_ref)}")
torch.testing.assert_close(Yt_ref, Yt_test, atol=0.0, rtol=tolerance)
print("pass.")

# TODO bias
# Y_test = mxfp8_matmul_NT(
#     Xq, scaleX,
#     Wq, scaleW,
#     out_dtype)    
# Y_ref = torch.addmm(b, X, W.T) 
# print(f"max element error: {max_error_element(Y_test, Y_ref)}")
# assert_allclose(Y_test, Y_ref, atol=tolerance)
torch.max()
# no bias
b = None
Yt_test = mxfp8_matmul_NT(
    Wq, scaleW,
    Xq, scaleX,
    out_dtype)

torch.testing.assert_close(Yt_ref, Yt_test, atol=0.0, rtol=5e-2)
# print(f"max element error: {max_error_element(Yt_test, Yt_ref)}")
# assert_allclose(Yt_test, Yt_ref, atol=tolerance)

# Gemm 2, grad X -----------------------------------------------------
Wq,      scaleW_swizzled = q_mxfp8_colwise(W.to(gradY.dtype))
grad_Yq, scaleY_swizzled = q_mxfp8_rowwise(gradY)

gradX_test, scale_gradX_test = op.cublaslt_mm_mxfp8(
                TransMatAB.NN.value, 
                Dout_dtype,
                Wq, scaleW_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)
# gradX_ref = torch.mm(gradY,   W)
# print(f"max element error: {max_error_element(gradX_test, gradX_ref)}")
# assert_allclose(gradX_test, gradX_ref, atol=tolerance)

# Gemm 3, grad W -----------------------------------------------------
Xq,      scaleX_swizzled = q_mxfp8_colwise(X.to(gradY.dtype))
grad_Yq, scaleY_swizzled = q_mxfp8_colwise(gradY)
gradW_test, scale_gradW_test = op.cublaslt_mm_mxfp8(
                TransMatAB.NT.value, 
                Dout_dtype,
                Xq, scaleX_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)
# gradW_ref = torch.mm(gradY.T, X)
# print(f"max element error: {max_error_element(gradW_test, gradW_ref)}")
# assert_allclose(gradW_test, gradW_ref, atol=tolerance)

if False:
    # CustomMatMul
    # CublasltLinearFunc
    dtype = torch.float32
    useB = False
    torch_linear = nn.Linear(IC, OC, bias=useB).to(device="cuda", dtype=dtype)
    custom_linear = CublasltMxfp8Linear(IC, OC, bias=useB).to(device="cuda", dtype=dtype)
    custom_linear.load_state_dict(torch_linear.state_dict())


    # x = torch.randn(B, L, IC).to(device="cuda", dtype=dtype)
    x_test = cuda_rand_tensor((B*L, IC), dtype=dtype)
    x_ref = x_test.clone() # we need another copy, otherwise, backward gonna accumulate on same tensor
    x_test.requires_grad = True
    x_ref.requires_grad = True

    y_test = custom_linear(x_test)
    y_ref = torch_linear(x_ref)

    print(f"max element error: {max_error_element(y_test, y_ref)}")
    # assert_allclose(y_test, y_ref, atol=tolerance)

    y_test.sum().backward()
    y_ref.sum().backward()

    print(f"max element error: {max_error_element(torch_linear.weight.grad, custom_linear.weight.grad)}")
    # assert_allclose(torch_linear.weight.grad, custom_linear.weight.grad, atol=tolerance)

    print(f"max element error: {max_error_element(x_test.grad, x_ref.grad)}")
    # assert_allclose(x_test.grad, x_ref.grad, atol=tolerance)
print("joto")

