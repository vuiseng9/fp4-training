import torch
from torchao.quantization.quant_primitives import (
    _choose_qparams_affine_float8, _quantize_affine_float8
)
from .linear import CustomLinear
import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

def quantize_f8(x, f8type):
    # per literature
    # for tensor wide f8 quantization
    # use torch.float8_e4m3fn for forward
    # use torch.float8_e5m2 for backward
    scale = _choose_qparams_affine_float8(x, block_size=[],
                                          float8_dtype=f8type)
    x_f8 = _quantize_affine_float8(x, scale, float8_dtype=f8type)
    return x_f8, scale

def f8mm(a, b, scale_a, scale_b, bias=None):
    # mat of shape [128, 16]
    # mat.stride() -> (16, 1) row major
    # mat.stride() -> (1, 128) col major
    def col_major(mat):
        if mat.stride()[0] > mat.stride()[1]:
            return mat.t().contiguous().t()
        return mat

    return torch._scaled_mm(a.contiguous(), col_major(b), scale_a, scale_b, bias=bias, out_dtype=torch.bfloat16)


class TorchFloat8MatMul(torch.autograd.Function):
    """
    Float8 matrix multiplication using torch._scaled_mm
    we quantize dynamically, tensor-wide for both input, 
    """

    @staticmethod
    def forward(ctx, X, W, b=None):
        # using X, W instead of generic A & B for easy correspondence to Linear layer
        # assume W following layout of nn.Linear, i.e. OCxIC
        if X.ndim != 2 or W.ndim != 2:
            raise ValueError("Expected 2D inputs")

        X_f8, scale_X = quantize_f8(X, f8type=torch.float8_e4m3fn)
        W_f8, scale_W = quantize_f8(W, f8type=torch.float8_e4m3fn)

        # out_dtype must be torch.bfloat16, we only upcast after that if needed
        if b is not None:
            Y = f8mm(X_f8, W_f8.T, scale_X, scale_W, b.to(torch.bfloat16)).to(dtype=X.dtype)
        else:
            Y = f8mm(X_f8, W_f8.T, scale_X, scale_W).to(dtype=X.dtype)

        ctx.save_for_backward(X, W) 
        # we can't save b if it is none 
        # but b is also not needed when it is used, constant -> zero in derivative
        ctx.has_bias = b is not None

        return Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors

        assert X.is_contiguous(), "X must be contiguous, they are by default, find out why it is not"
        assert W.is_contiguous(), "W must be contiguous, they are by default, find out why it is n"
        warnings.warn(f"grad_Y.is_contiguous()={grad_Y.is_contiguous()}, it is expected to be non-contiguous, stride(0,0), to contiguous()")
        grad_Y = grad_Y.contiguous()

        grad_X = grad_W = grad_b = None

        # https://github.com/pytorch/pytorch/issues/132005

        # tensor-wide scaler, no layout constraint, quantize grad_Y in common area
        grad_Y_f8, scale_grad_Y = quantize_f8(grad_Y, f8type=torch.float8_e4m3fn)

        if ctx.needs_input_grad[0] is True:
            W_f8, scale_W = quantize_f8(W, f8type=torch.float8_e4m3fn)
            grad_X = f8mm(grad_Y_f8, W_f8, scale_grad_Y, scale_W).to(dtype=grad_Y.dtype)

        if ctx.needs_input_grad[1] is True:
            X_f8, scale_X = quantize_f8(X, f8type=torch.float8_e4m3fn)
            grad_W = f8mm(grad_Y_f8.T, X_f8, scale_grad_Y, scale_X).to(dtype=grad_Y.dtype)

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0) # Original bias shape (OC,)

        return grad_X, grad_W, grad_b


class TorchFloat8Linear(CustomLinear):
    """
    Custom Linear layer using TorchFloat8MatMul for forward and backward pass.
    This is a drop-in replacement for nn.Linear with per-tensor float8 quantization.
    """

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  TorchFloat8MatMul.apply(input, self.weight, self.bias)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out