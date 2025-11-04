import warnings
warnings.simplefilter("once", UserWarning)   # warn once per callsite

from collections import OrderedDict

import torch
import backend.xops
op = torch.ops.xops
from torch.amp import custom_fwd, custom_bwd, autocast

from .custom import CustomLinear
from .cublaslt import TransAB
from ..quantize import q_nvfp4_rowwise


class Nvfp4Matmul(torch.autograd.Function):
    """
    NVFP4 Linear Autograd function that calls
    pytorch extended op cublaslt_mm_nvfp4.
    Note: only TN layout is supported cublaslt nvfp4,
            for now, we brute force it by transposing,
            very inefficient.
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
            # brute force for TN layout, nvfp4 cublaslt only supports one layout
            Wt = W.to(grad_Y.dtype).T.contiguous()

            Wtq,     scaleWt_swizzled = ctx.quant['2A'](Wt)
            grad_Yq, scaleY_swizzled  = ctx.quant['2B'](grad_Y)
            
            grad_X, _ = op.cublaslt_mm_nvfp4(
                TransAB.TN.value, 
                grad_Y.dtype,
                Wtq,     scaleWt_swizzled,
                grad_Yq, scaleY_swizzled,
                None)

        # gemm 3
        if ctx.needs_input_grad[1] is True:
            # brute force for TN layout, nvfp4 cublaslt only supports one layout
            Xt = X.to(grad_Y.dtype).T.contiguous()
            grad_Yt = grad_Y.T.contiguous()

            Xtq,      scaleXt_swizzled = ctx.quant['3A'](Xt)
            grad_Ytq, scaleYt_swizzled = ctx.quant['3B'](grad_Yt)

            grad_W, _ = op.cublaslt_mm_nvfp4(
                TransAB.TN.value,
                grad_Y.dtype,
                Xtq,      scaleXt_swizzled, 
                grad_Ytq, scaleYt_swizzled,
                None)

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None


class CublasltNvfp4Linear(CustomLinear):
    """
    NVFP4 Linear Layer: Quantization using Microxcaling,
    Matmul using cublaslt nvfp4 gemm wrapped in Nvfp4MatMul autograd above.
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
        # always rowwise because always TN
        self.quantizers['1A'] = q_nvfp4_rowwise # W/IC
        self.quantizers['1B'] = q_nvfp4_rowwise # X/IC
        self.quantizers['2A'] = q_nvfp4_rowwise # Wt/OC
        self.quantizers['2B'] = q_nvfp4_rowwise # dY/OC
        self.quantizers['3A'] = q_nvfp4_rowwise # X/N
        self.quantizers['3B'] = q_nvfp4_rowwise # dYt/N

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  Nvfp4Matmul.apply(input, self.weight, self.bias, self.quantizers)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
