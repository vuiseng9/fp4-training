import torch
from mxfp.mx.mx_ops import quantize_mx_op
from mxfp.mx.specs import MxSpecs

from functools import partial
import math
from .swizzle import swizzle_rowwise_scale, swizzle_colwise_scale

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

_q_mxfp8_rowwise = partial(quantize_mx_op, mx_specs=mxfp8_spec, elem_format='fp8_e4m3', block_size=mxfp8_spec['block_size'], axes=[-1], expand_and_reshape=True, return_quantized=True)
_q_mxfp8_colwise = partial(quantize_mx_op, mx_specs=mxfp8_spec, elem_format='fp8_e4m3', block_size=mxfp8_spec['block_size'], axes=[-2], expand_and_reshape=True, return_quantized=True)

def q_mxfp8_rowwise(tensor):
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D for mxfp8 quantization")

    orig_shape = tensor.shape
    q_tensor, scale = _q_mxfp8_rowwise(tensor)
    q_tensor = q_tensor.view(orig_shape).to(torch.float8_e4m3fn).contiguous().view(torch.uint8) # _q_mxfp8_rowwise is in per mx block, need to flatten
    
    # biasing scale
    scale += 127
    scale = scale.to(torch.uint8)
    scale = swizzle_rowwise_scale(scale)
    return q_tensor, scale

def q_mxfp8_colwise(tensor):
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D for mxfp8 quantization")

    orig_shape = tensor.shape
    q_tensor, scale = _q_mxfp8_colwise(tensor)
    q_tensor = q_tensor.view(orig_shape).to(torch.float8_e4m3fn).contiguous().view(torch.uint8) # _q_mxfp8_colwise is in per mx block, need to flatten

    # biasing scale
    scale += 127
    scale = scale.to(torch.uint8)
    swiz_scale = swizzle_colwise_scale(scale)
    return q_tensor, swiz_scale

if __name__ == "__main__":
    a = torch.randn(2, 64).cuda()
    fqa = fq_mxfp8_rowwise(a)
    
    aq, scale_a = q_mxfp8_rowwise(a)
    expanded_scale_a = scale_a.repeat_interleave(32, dim=1)
    reconstructed_a = aq * (2**expanded_scale_a)

    assert (fqa == reconstructed_a).all().item(), "Debug"
    print(f"a: max quantized error: {(reconstructed_a - a).abs().max().item():.5f}")

    b = torch.randn(64, 16).cuda()
    fqb = fq_mxfp8_colwise(b)

    bq, scale_b = q_mxfp8_colwise(b)
    expanded_scale_b = scale_b.repeat_interleave(32, dim=0)
    reconstructed_b = bq * (2**expanded_scale_b)

    assert (fqb == reconstructed_b).all().item(), "Debug"
    print(f"b: max quantized error: {(reconstructed_b - b).abs().max().item():.5f}")

    print("end.")