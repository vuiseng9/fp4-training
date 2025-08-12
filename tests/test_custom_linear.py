import pytest
import torch
import torch.nn as nn
from custom import CustomLinear, CudaMMLinear, CuBlasltMMLinear

LINEAR_TOLERANCES = [
    # (class, atol)
    (CustomLinear, 1e-5),
    (CudaMMLinear, 1e-5),
    (CuBlasltMMLinear, 1e-3)
]

LINEAR_TESTLIST = list(map(lambda t: t[0], LINEAR_TOLERANCES))

dtype_label = {
    torch.float32: "f32",
    torch.bfloat16: "bf16"
}

class TestCustomizedLinear:

    @pytest.mark.parametrize("use_bias", [True, False], ids=lambda x: f"bias-{x}")
    @pytest.mark.parametrize("oc", [5, 15, 30], ids=lambda x: f"{x}.oc")
    @pytest.mark.parametrize("ic", [10, 20, 50], ids=lambda x: f"{x}.ic")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=lambda x: dtype_label.get(x))
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("constructor", LINEAR_TESTLIST)
    def test_construction(self, ic, oc, use_bias, dtype, device, constructor):
        layer = constructor(ic, oc, bias=use_bias).to(device=device, dtype=dtype)
        assert layer.in_features == ic
        assert layer.out_features == oc
        assert layer.weight.shape == (oc, ic)
        assert layer.weight.dtype == dtype
        assert layer.weight.is_cuda == (device == "cuda")
        assert layer.bias.shape == (oc,) if use_bias else layer.bias is None
        assert layer.bias.dtype == dtype if use_bias else layer.bias is None
        assert layer.bias.is_cuda == (device == "cuda") if use_bias else True

    @pytest.mark.parametrize("use_bias", [True, False], ids=lambda x: f"bias-{x}")
    @pytest.mark.parametrize("oc", [5, 15, 30], ids=lambda x: f"{x}.oc")
    @pytest.mark.parametrize("ic", [10, 20, 50], ids=lambda x: f"{x}.ic")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=lambda x: dtype_label.get(x))
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("constructor", LINEAR_TESTLIST)
    def test_construction_from_linear(self, ic, oc, use_bias, dtype, device, constructor):
        torch_linear = nn.Linear(ic, oc, bias=use_bias).to(device=device, dtype=dtype)
        custom_linear = constructor.from_linear(torch_linear)

        assert custom_linear.in_features == torch_linear.in_features
        assert custom_linear.out_features == torch_linear.out_features
        assert custom_linear.weight.shape == torch_linear.weight.shape
        assert custom_linear.weight.dtype == torch_linear.weight.dtype
        assert custom_linear.weight.is_cuda == torch_linear.weight.is_cuda
        
        assert (custom_linear.weight == torch_linear.weight).all()

        if use_bias:
            assert custom_linear.bias.shape == torch_linear.bias.shape 
            assert custom_linear.bias.dtype == torch_linear.bias.dtype 
            assert custom_linear.bias.is_cuda == torch_linear.bias.is_cuda

            assert (custom_linear.bias == torch_linear.bias).all()

    @pytest.mark.parametrize("use_bias", [True, False], ids=lambda x: "xbias" if not x else "bias")
    @pytest.mark.parametrize("oc", [16, 32, 128], ids=lambda x: f"{x}.oc")
    @pytest.mark.parametrize("ic", [16, 32, 256], ids=lambda x: f"{x}.ic")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=lambda x: dtype_label.get(x))
    @pytest.mark.parametrize("constructor_tol", LINEAR_TOLERANCES, ids=lambda x: x[0].__name__ + f"-{x[1]}")
    def test_fwd_bwd_cuda(self, ic, oc, use_bias, dtype, constructor_tol):
        constructor, atol = constructor_tol
        torch_linear = nn.Linear(ic, oc, bias=use_bias).to(device="cuda", dtype=dtype)
        custom_linear = constructor.from_linear(torch_linear)

        B = 4
        L = 17
        # 2D inputs
        x = torch.randn(B, ic).to(device="cuda", dtype=dtype)
        y = custom_linear(x)
        ref_y = torch_linear(x)

        assert y.shape == (B, oc)
        assert y.dtype == dtype
        assert y.is_cuda
        assert torch.allclose(y, ref_y, atol=atol)

        # 3D inputs
        x = torch.randn(B, L, ic).to(device="cuda", dtype=dtype)
        ref_x = x.clone() # we need another copy, otherwise, backward gonna accumulate on same tensor
        x.requires_grad = True
        ref_x.requires_grad = True

        y = custom_linear(x)
        ref_y = torch_linear(ref_x)

        assert y.shape == (B, L, oc)
        assert y.dtype == dtype
        assert y.is_cuda
        assert torch.allclose(y, ref_y, atol=atol)

        # Backward pass
        y.sum().backward()
        ref_y.sum().backward()

        assert torch.allclose(custom_linear.weight.grad, torch_linear.weight.grad, atol=atol)
        assert torch.allclose(x.grad, ref_x.grad, atol=atol)
        
        if use_bias:
            assert torch.allclose(custom_linear.bias.grad, torch_linear.bias.grad, atol=atol)

