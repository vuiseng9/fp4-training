import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

from enum import IntEnum

import torch
import backend.xops
op = torch.ops.xops
from torch.amp import custom_fwd, custom_bwd

from .custom import CustomLinear


class TransAB(IntEnum):
    TN = 0
    NN = 1
    NT = 2

class CublasltMMFunc(torch.autograd.Function):
    """
    Autograd Linear function that calls
    pytorch extended op cublaslt_mm_fp32_or_bf16.
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16) 
    def forward(ctx, X, W, b=None):
        # no shape checking as it is handled at backend function

        # gemm 1
        Y = op.cublaslt_mm_fp32_or_bf16(TransAB.TN, W, X, b)  # b can be None or vector

        ctx.save_for_backward(X, W) 
        ctx.has_bias = b is not None
        return Y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        
        assert X.is_contiguous(), "X must be contiguous, they are by default, find out why it is not"
        assert W.is_contiguous(), "W must be contiguous, they are by default, find out why it is n"
        # warnings.warn(f"grad_Y.is_contiguous()={grad_Y.is_contiguous()}, it is expected to be non-contiguous, stride(0,0), to contiguous()")
        grad_Y = grad_Y.contiguous()
        
        grad_X = grad_W = grad_b = None

        # gemm 2
        if ctx.needs_input_grad[0] is True:
            grad_X = op.cublaslt_mm_fp32_or_bf16(TransAB.NN, W, grad_Y, None)

        # gemm 3
        if ctx.needs_input_grad[1] is True:
            grad_W = op.cublaslt_mm_fp32_or_bf16(TransAB.NT, X, grad_Y, None)

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0) # Original bias shape (OC,)

        return grad_X, grad_W, grad_b

class CublasltLinear(CustomLinear):
    """
    Custom linear layer using cublasLtMatmul.
    """
    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  CublasltMMFunc.apply(input, self.weight, self.bias)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out