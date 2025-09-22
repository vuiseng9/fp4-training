import torch
from torch import nn
import backend.xops
from custom import CublasltMxfp8Linear

op = torch.ops.xops
from custom.quantize import q_nvfp4_rowwise, q_nvfp4_colwise
from custom.mxfp8_linear import TransMatAB

def cuda_rand_tensor(shape, dtype):
    return torch.rand(*shape).cuda().to(dtype)

def max_error_element(x, ref):
    return (x-ref).abs().max()

def assert_allclose(x, ref, atol=1e-5):
    assert torch.allclose(x, ref, atol=atol), f"max error {max_error_element(x, ref)} > {atol}"

def noraise_allclose(*args, **kwargs):
    try:
        # Attempt the assertion
        torch.testing.assert_close(*args, **kwargs)
        print("✅ Tensors are close!")
    except AssertionError as e:
        # If it fails, catch the error and print its message
        print("❌ Tensors are NOT close. See comparison below:")
        print(e)

B = 4
L = 16
IC = 128
OC = 128

# this script now harden for bfloat16 inputs, hardcoded for f8_e4m3 dout_type
# CUBLASLT_LOG_LEVEL=2 to verify dout_type (Ddesc) in cublas logs
input_dtype = torch.bfloat16
Dout_dtype = torch.bfloat16
tolerance = 1e3
# X = torch.rand(B*L, IC).cuda().to(torch.float32)
# W = torch.rand(OC, IC).cuda().to(torch.float32)

X = cuda_rand_tensor((B*L, IC), dtype=input_dtype)
W = cuda_rand_tensor((OC, IC), dtype=input_dtype)
gradY = cuda_rand_tensor((B*L, OC), dtype=input_dtype)
b = cuda_rand_tensor((OC,), dtype=input_dtype)

# Gemm 1, fwd -----------------------------------------------------
Wq, scaleW_swizzled = q_nvfp4_rowwise(W)
Xq, scaleX_swizzled = q_nvfp4_rowwise(X)

Y_test, scaleY_test = op.cublaslt_mm_nvfp4(
    TransMatAB.TN.value,
    Dout_dtype, 
    Wq, scaleW_swizzled, 
    Xq, scaleX_swizzled,
    b)
Y_ref = torch.addmm(b, X, W.T) 
noraise_allclose(Y_test, Y_ref, atol=0.00, rtol=0.25)

# no bias
b = None
Y_test, scaleY_test = op.cublaslt_mm_nvfp4(
    TransMatAB.TN.value,
    Dout_dtype, 
    Wq, scaleW_swizzled, 
    Xq, scaleX_swizzled,
    b)
Y_ref = torch.mm(X, W.T)
noraise_allclose(Y_test, Y_ref, atol=0.00, rtol=0.25)

print("joto")


# Gemm 2, grad X -----------------------------------------------------
Wq,      scaleW_swizzled = q_nvfp4_colwise(W.to(gradY.dtype))
grad_Yq, scaleY_swizzled = q_nvfp4_rowwise(gradY)

gradX_test, scale_gradX_test = op.cublaslt_mm_nvfp4(
                TransMatAB.NN.value, 
                Dout_dtype,
                Wq, scaleW_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)
# gradX_ref = torch.mm(gradY,   W)
# print(f"max element error: {max_error_element(gradX_test, gradX_ref)}")
# assert_allclose(gradX_test, gradX_ref, atol=tolerance)

# Gemm 3, grad W -----------------------------------------------------
Xq,      scaleX_swizzled = q_nvfp4_colwise(X.to(gradY.dtype))
grad_Yq, scaleY_swizzled = q_nvfp4_colwise(gradY)
gradW_test, scale_gradW_test = op.cublaslt_mm_nvfp4(
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

