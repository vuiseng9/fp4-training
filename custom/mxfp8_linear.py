from enum import Enum
import torch
from torch.amp import custom_fwd, custom_bwd, autocast

import backend.xops
op = torch.ops.xops

import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

from functools import partial
from collections import OrderedDict

from .linear import CustomLinear
from .quantize import q_mxfp8_rowwise, q_mxfp8_colwise

def raise_if_not_contiguous(tensor, name):
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, but got shape {tensor.shape} and stride {tensor.stride()}")

# TODO: following Enums are manually mapped, requiring manual changes if cpp side is modified.
#    A better implementation is single source from cpp side.
class TransMatAB(Enum):
    TN = 0
    NN = 1
    NT = 2

class DoutType(Enum):
    F32 = torch.float32
    BF16 = torch.bfloat16
    F8 = torch.float8_e4m3fn
class Mxfp8MatMul(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  
    # makes *incoming* tensors BF16, meaning X, W, b will be casted to BF16 if autocast is enabled. 
    # Implication input to quantization is bf16. Stick to this for now, need deeper understanding of autocast.
    def forward(ctx, X, W, b, quant: OrderedDict):
        Wq, scaleW_swizzled = quant['1A'](W)
        Xq, scaleX_swizzled = quant['1B'](X)

        # Call the CUDA extension with autocast disabled to avoid any hidden casts (because it has been casted)
        with autocast(device_type="cuda", enabled=False):
            Y = op.cublaslt_mm_mxfp8(
                TransMatAB.TN.value,
                X.dtype,
                Wq, scaleW_swizzled, 
                Xq, scaleX_swizzled,
                b)

        ctx.save_for_backward(X, W)
        ctx.quant = quant
        ctx.has_bias = b is not None
        return Y
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors
        assert X.is_contiguous(), "X must be contiguous, they are by default, find out why it is not"
        assert W.is_contiguous(), "W must be contiguous, they are by default, find out why it is not"
        
        warnings.warn(f"grad_Y.is_contiguous()={grad_Y.is_contiguous()}, it is expected to be non-contiguous, stride(0,0), to contiguous()")
        grad_Y = grad_Y.contiguous()      

        grad_X = grad_W = grad_b = None

        if ctx.needs_input_grad[0] is True:
            Wq,      scaleW_swizzled = ctx.quant['2A'](W.to(grad_Y.dtype))
            grad_Yq, scaleY_swizzled = ctx.quant['2B'](grad_Y)
            
            with autocast(device_type="cuda", enabled=False):
                grad_X = op.cublaslt_mm_mxfp8(
                    TransMatAB.NN.value, 
                    grad_Y.dtype,
                    Wq, scaleW_swizzled, 
                    grad_Yq, scaleY_swizzled,
                    None)

        if ctx.needs_input_grad[1] is True:
            Xq,      scaleX_swizzled = ctx.quant['3A'](X.to(grad_Y.dtype))
            grad_Yq, scaleY_swizzled = ctx.quant['3B'](grad_Y)
            
            grad_W = op.cublaslt_mm_mxfp8(
                TransMatAB.NT.value, 
                grad_Y.dtype,
                Xq, scaleX_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)


        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None


class CublasltMxfp8Linear(CustomLinear):
    """
    Linear Layer with fwd and bwd gemm simulated in mxfp8
    using offical mxfp emulation kit
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quantizers()

    @classmethod
    def from_linear(cls, linear_layer):
        linear = super().from_linear(linear_layer)
        linear._init_quantizers()
        return linear
    
    def _init_quantizers(self):
        self.quantizers = OrderedDict()
        self.quantizers['1A'] = q_mxfp8_rowwise # W/IC
        self.quantizers['1B'] = q_mxfp8_rowwise # X/IC
        self.quantizers['2A'] = q_mxfp8_colwise # W/OC
        self.quantizers['2B'] = q_mxfp8_rowwise # dY/OC
        self.quantizers['3A'] = q_mxfp8_colwise # X/N
        self.quantizers['3B'] = q_mxfp8_colwise # dY/N

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  Mxfp8MatMul.apply(input, self.weight, self.bias, self.quantizers)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
