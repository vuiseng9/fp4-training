import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

from collections import OrderedDict

import torch
import backend.xops
op = torch.ops.xops
from torch.amp import custom_fwd, custom_bwd, autocast

from .cublaslt import TransAB
from .custom import CustomLinear
from ..quantize import q_nvfp4_rowwise
from ..quantize import q_mxfp8_rowwise, q_mxfp8_colwise


class FwdNvfp4BwdMxfp8Matmul(torch.autograd.Function):
    """
    NVFP4 Forward (gemm 1) and MXFP8 Backward (gemm 2 & 3)
    using cublaslt_mm_mxfp8, cublaslt_mm_nvfp4
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  
    def forward(ctx, X, W, b, quant: OrderedDict):
        Wq, scaleW_swizzled = quant['1A'](W)
        Xq, scaleX_swizzled = quant['1B'](X)

        # gemm 1
        Y, _ = op.cublaslt_mm_nvfp4(
            TransAB.TN.value,
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
        
        # warnings.warn(f"grad_Y.is_contiguous()={grad_Y.is_contiguous()}, it is expected to be non-contiguous, stride(0,0), to contiguous()")
        grad_Y = grad_Y.contiguous()      

        grad_X = grad_W = grad_b = None

        # gemm 2
        if ctx.needs_input_grad[0] is True:
            Wq,      scaleW_swizzled = ctx.quant['2A'](W.to(grad_Y.dtype))
            grad_Yq, scaleY_swizzled = ctx.quant['2B'](grad_Y)

            grad_X, _ = op.cublaslt_mm_mxfp8(
                TransAB.NN.value, 
                grad_Y.dtype,
                Wq, scaleW_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)

        # gemm 3
        if ctx.needs_input_grad[1] is True:
            Xq,      scaleX_swizzled = ctx.quant['3A'](X.to(grad_Y.dtype))
            grad_Yq, scaleY_swizzled = ctx.quant['3B'](grad_Y)
            
            grad_W, _ = op.cublaslt_mm_mxfp8(
                TransAB.NT.value, 
                grad_Y.dtype,
                Xq, scaleX_swizzled, 
                grad_Yq, scaleY_swizzled,
                None)

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None


class CublasltFwdNvfp4BwdMxfp8Linear(CustomLinear):
    """
    Linear Layer with 
        forward using cublasLt nvfp4 gemm
        backward using cublasLt mxfp8 gemm.
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
        self.quantizers['1A'] = q_nvfp4_rowwise # W/IC
        self.quantizers['1B'] = q_nvfp4_rowwise # X/IC
        self.quantizers['2A'] = q_mxfp8_colwise # W/OC
        self.quantizers['2B'] = q_mxfp8_rowwise # dY/OC
        self.quantizers['3A'] = q_mxfp8_colwise # X/N
        self.quantizers['3B'] = q_mxfp8_colwise # dY/N

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  FwdNvfp4BwdMxfp8Matmul.apply(input, self.weight, self.bias, self.quantizers)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
