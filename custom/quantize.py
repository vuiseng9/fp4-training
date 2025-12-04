import torch
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.specs import MxSpecs

from functools import partial
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

# NVFP4 Quantization function
E2_MAX = 2 # unbiased exponent of E2M1, max
E2_MIN = 0 # unbiased exponent of E2M1, min
E2M1_NORM_MAX =  6.0
E4M3_NORM_MAX =  448.0
E4M3_SUBNORM_MIN = 2**-9
def quantize_nvfp4(tensor, rowwise=True, simulated=True, stochastic_rounding=False):
    tensor = tensor.to(torch.float32)
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D for nvfp4 quantization")
    orishape = tensor.shape
    oridtype = tensor.dtype

    if rowwise is True and orishape[1] % 16 != 0:
        raise ValueError("For nvfp4 row-wise quantization, the number of columns must be a multiple of 16")
    if rowwise is False and orishape[0] % 16 != 0:
        raise ValueError("For nvfp4 column-wise quantization, the number of rows must be a multiple of 16")

    if rowwise is False: # colwise
        tensor = tensor.T
    
    shape = tensor.shape # IMPORTANT, this shape is post-transpose, we reshape back to this and transpose back at the end if needed
    nvblocked = tensor.reshape(shape[0], -1, 16)
    
    absmax = nvblocked.abs().max(-1, keepdim=True).values
    scales = (absmax/E2M1_NORM_MAX).clamp(E4M3_SUBNORM_MIN, E4M3_NORM_MAX)
    
    scaled_nvblocked = (nvblocked / scales)
    # q_nvblocked is in original dtype, need to convert to fp4
    sign = scaled_nvblocked.sign()
    scaled_nvblocked = scaled_nvblocked.abs()
    eps = torch.finfo(scaled_nvblocked.dtype).eps    
    # find E2 (note that we avoid redundant bias by baking in the bias)
    e = torch.floor(torch.log2(scaled_nvblocked.clamp(min=eps))).clamp(E2_MIN, E2_MAX)
    # find M1
    m = (scaled_nvblocked / (2**e))
    if stochastic_rounding is True:
        noise = torch.rand_like(m) - 0.5
        m = torch.round( m*2 + noise) / 2     
    else:
        m = torch.round( m*2 ) / 2 # nearest rounding # Round Tie to Even

    q_nvblocked = sign * (2**e) * m
    # scales, q_nvblocked calculation complete, shapes are not blocked

    if simulated is True:
        reconstructed_tensor = (scales * q_nvblocked).reshape(shape)
        if rowwise is False: # colwise
            return reconstructed_tensor.T
        return reconstructed_tensor
    else:
        q_tensor = q_nvblocked.reshape(shape).to(oridtype)
        scales = scales.squeeze().to(torch.float8_e4m3fn)
        if rowwise is False: # colwise
            return q_tensor.T, scales.T
        return q_tensor, scales

quant_fn_rowwise = partial(quantize_nvfp4, rowwise=True, simulated=False)
def q_nvfp4_rowwise(tensor, sr=False):
    q_tensor, scales = quant_fn_rowwise(tensor, stochastic_rounding=sr)
    # we only pack q_tensor in cpp implementation using cuda util function
    # scales will be swizzled
    scales = swizzle_rowwise_scale(scales)
    return q_tensor.contiguous(), scales.view(torch.uint8)
    
quant_fn_colwise = partial(quantize_nvfp4, rowwise=False, simulated=False)
def q_nvfp4_colwise(tensor, sr=False):
    q_tensor, scales = quant_fn_colwise(tensor, stochastic_rounding=sr)
    # we only pack q_tensor in cpp implementation using cuda util function
    # scales will be swizzled
    scales = swizzle_colwise_scale(scales)
    return q_tensor.contiguous(), scales.view(torch.uint8)


if __name__ == "__main__":
    a = torch.randn(2, 64).cuda()
    fp4a, scale_fp4a = q_nvfp4_rowwise(a)

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