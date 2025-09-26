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
from .quantize import q_nvfp4_rowwise, q_nvfp4_colwise, swizzle_rowwise_scale, swizzle_colwise_scale
from .quantize import q_mxfp8_rowwise, q_mxfp8_colwise

from .mxfp8_linear import TransMatAB

from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
# Note: require def changes in nvfp4_tensor.py
# class NVFP4Quantizer(MXFP8Quantizer):
#     def __init__(self, rowwise, columnwise):
#         super().__init__(TE_DType.kFloat4E2M1, rowwise=rowwise, columnwise=columnwise)

rowwise_quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
colwise_quantizer = NVFP4Quantizer(rowwise=False, columnwise=True)

def te_q_nvfp4_rowwise(tensor):
    qtensor = rowwise_quantizer(tensor)
    qdata = qtensor._rowwise_data
    scale = qtensor._rowwise_scale_inv
    return qdata, swizzle_rowwise_scale(scale)

# unused due to TN Layout
def te_q_nvfp4_colwise(tensor):
    qtensor = colwise_quantizer(tensor)
    qdata = qtensor._columnwise_data
    scale = qtensor._columnwise_scale_inv
    return qdata, swizzle_colwise_scale(scale)

def raise_if_not_contiguous(tensor, name):
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, but got shape {tensor.shape} and stride {tensor.stride()}")


class Nvfp4Matmul(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  
    # makes *incoming* tensors BF16, meaning X, W, b will be casted to BF16 if autocast is enabled. 
    # Implication input to quantization is bf16. Stick to this for now, need deeper understanding of autocast.
    def forward(ctx, X, W, b, quant: OrderedDict):
        Wq, scaleW_swizzled = quant['1A'](W)
        Xq, scaleX_swizzled = quant['1B'](X)

        # Call the CUDA extension with autocast disabled to avoid any hidden casts (because it has been casted)
        with autocast(device_type="cuda", enabled=False):
            Y, _ = op.cublaslt_mm_nvfp4_packedAB(
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
            # brute force for TN layout
            Wt = W.to(grad_Y.dtype).T.contiguous()

            Wtq,     scaleWt_swizzled = ctx.quant['2A'](Wt)
            grad_Yq, scaleY_swizzled  = ctx.quant['2B'](grad_Y)
            
            with autocast(device_type="cuda", enabled=False):
                grad_X, _ = op.cublaslt_mm_nvfp4_packedAB(
                    TransMatAB.TN.value, 
                    grad_Y.dtype,
                    Wtq,     scaleWt_swizzled,
                    grad_Yq, scaleY_swizzled,
                    None)

        if ctx.needs_input_grad[1] is True:
            # brute force for TN layout
            Xt = X.to(grad_Y.dtype).T.contiguous()
            grad_Yt = grad_Y.T.contiguous()

            Xtq,      scaleXt_swizzled = ctx.quant['3A'](Xt)
            grad_Ytq, scaleYt_swizzled = ctx.quant['3B'](grad_Yt)

            grad_W, _ = op.cublaslt_mm_nvfp4_packedAB(
                TransMatAB.TN.value,
                grad_Y.dtype,
                Xtq,      scaleXt_swizzled, 
                grad_Ytq, scaleYt_swizzled,
                None)


        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None


class CublasltNvfp4Linear(CustomLinear):
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
        self.quantizers['1A'] = te_q_nvfp4_rowwise # W/IC
        self.quantizers['1B'] = te_q_nvfp4_rowwise # X/IC
        self.quantizers['2A'] = te_q_nvfp4_rowwise # Wt/OC
        self.quantizers['2B'] = te_q_nvfp4_rowwise # dY/OC
        self.quantizers['3A'] = te_q_nvfp4_rowwise # X/N
        self.quantizers['3B'] = te_q_nvfp4_rowwise # dYt/N

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  Nvfp4Matmul.apply(input, self.weight, self.bias, self.quantizers)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
