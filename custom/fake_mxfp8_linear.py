import torch
import backend.xops
op = torch.ops.xops

from mxfp.mx.mx_ops import quantize_mx_op
from mxfp.mx.specs import MxSpecs

from functools import partial
from collections import OrderedDict

from .linear import CustomLinear

mxfp8_spec = MxSpecs({
    'scale_bits': 8,           # Bits for shared exponent/scale
    'block_size': 32,           # Block size for shared scaling
    'shared_exp_method': 'max',         # Use max value in block for scaling
    'round': 'nearest',                 # Rounding mode
    'mx_flush_fp32_subnorms': False,    # Don't flush subnormal FP32 values
    'custom_cuda': False,               # Use PyTorch implementation
})

fq_mxfp8_rowwise = partial(quantize_mx_op, mx_specs=mxfp8_spec, elem_format='fp8_e4m3', block_size=mxfp8_spec['block_size'], axes=[-1])
fq_mxfp8_colwise = partial(quantize_mx_op, mx_specs=mxfp8_spec, elem_format='fp8_e4m3', block_size=mxfp8_spec['block_size'], axes=[-2])

class FakeMxfp8MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, quant: OrderedDict, use_cublaslt=False):
        fqX = quant[0](X)
        fqW = quant[1](W)
        
        if use_cublaslt:
            Y = op.cublaslt_matmul_bias_epilogue(fqW, fqX.T, b)
        else:
            # attempt to simulate A=W, B=X.T
            Y = torch.matmul(fqW, fqX.T).T

        ctx.save_for_backward(X, W)
        ctx.quant = quant
        ctx.has_bias = b is not None
        if b is not None and not use_cublaslt:
            return Y + b
        return Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors

        grad_X = grad_W = grad_b = None

        if ctx.needs_input_grad[0] is True:
            grad_X = torch.matmul(
                ctx.quant[2](grad_Y), 
                ctx.quant[3](W.to(grad_Y.dtype))
            )
        
        if ctx.needs_input_grad[1] is True:
            grad_W = torch.matmul(
                ctx.quant[4](grad_Y.view(-1, grad_Y.shape[-1])).T, # transpose after quantization
                ctx.quant[5](X.to(grad_Y.dtype))
            )

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None

class FakeMxfp8Linear(CustomLinear):
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
        self.quantizers[0] = fq_mxfp8_rowwise # X/IC
        self.quantizers[1] = fq_mxfp8_rowwise # W/IC
        self.quantizers[2] = fq_mxfp8_rowwise # dY/OC
        self.quantizers[3] = fq_mxfp8_colwise # W/OC
        self.quantizers[4] = fq_mxfp8_colwise # dY/N
        self.quantizers[5] = fq_mxfp8_colwise # X/N

    def forward(self, input):
        shapes = None
        if input.ndim > 2:
            shapes = input.shape
            input = input.view(-1, shapes[-1])
        out =  FakeMxfp8MatMul.apply(input, self.weight, self.bias, self.quantizers)

        if shapes is not None:
            out = out.view(shapes[:-1] + (self.out_features,))
        return out
