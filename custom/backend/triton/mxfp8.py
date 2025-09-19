import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from custom.quantize import _q_mxfp8_rowwise, _q_mxfp8_colwise
from functools import partial
# from ...mxfp8_linear import TransMatAB

def ceilto(x, div):
    return ((x + div - 1) // div) * div

def pad_to_multiple_of_128x4tile(scale_2d):
    M, Kdiv = scale_2d.shape
    if M % 128 == 0 and Kdiv % 4 == 0:
        return scale_2d
    M_pad   = ceilto(M, 128)
    Kdiv_pad= ceilto(Kdiv, 4)
    out = scale_2d.new_zeros((M_pad, Kdiv_pad))
    out[:M, :Kdiv] = scale_2d
    return out

def ceilto(n, block_dim):
    return ((n + block_dim - 1) // block_dim) * block_dim

def tile_matrix(tensor, ty, tx, return_tiled=True):
    # pad to tile boundary, return 4D tensor 
    if tensor.shape[0] % ty != 0 or tensor.shape[1] % tx != 0:
        padded = torch.zeros((ceilto(tensor.shape[0], ty), ceilto(tensor.shape[1], tx)), dtype=tensor.dtype, device=tensor.device)
        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
        tensor = padded
    
    if return_tiled is True:
        return tensor.view(tensor.shape[0]//ty, ty, tensor.shape[1]//tx, tx).permute(0,2,1,3).contiguous()
    return tensor

tile_128x4 = partial(tile_matrix, ty=128, tx=4)
tile_4x128 = partial(tile_matrix, ty=4, tx=128)

pad_to_multiple = partial(tile_matrix, return_tiled=False)

def q_mxfp8(tensor, rowwise=True):
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D for mxfp8 quantization")
    orig_shape = tensor.shape

    if rowwise:
        q_tensor, scale = _q_mxfp8_rowwise(tensor)
    else:
        q_tensor, scale = _q_mxfp8_colwise(tensor)

    q_tensor = q_tensor.view(orig_shape).to(torch.float8_e4m3fn).contiguous() # _q_mxfp8_rowwise is in per mx block, need to flatten
    
    # biasing scale
    scale += 127
    scale = scale.to(torch.uint8).contiguous()
    if rowwise:
        scale = tile_128x4(scale)
    else:
        scale = tile_4x128(scale)
    return q_tensor, scale

quantize_rowwise = partial(q_mxfp8, rowwise=True)
quantize_colwise = partial(q_mxfp8, rowwise=False)

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret

@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.bfloat16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


KERNELCFG = {
    'BLOCK_SIZE_M': 128, 
    'BLOCK_SIZE_N': 256, 
    'BLOCK_SIZE_K': 128, 
    'num_stages': 2, 
    'ELEM_PER_BYTE_A': 1, 
    'ELEM_PER_BYTE_B': 1, 
    'VEC_SIZE': 32
}


def mxfp8_matmul_NT(A, scaleA, B, scaleB, out_dtype):
    # A: (M, K) row-major, e4m3fn, M multiple of 128 
    # B: (N, K) row-major, e4m3fn, N multiple of 256
    # K multiple of 128
    # scaleA: (tiley, tilex, 128, 4), row-major uint8, i.e. tiles of 128x4
    # scaleB: (tiley, tilex, 128, 4), row-major uint8, i.e. tiles of 128x4
    # dtype_dst: torch.float32, torch.float16, torch.float8_e4m

    if out_dtype == torch.float32:
        dtype_dst_id = 0
    elif out_dtype == torch.bfloat16:
        dtype_dst_id = 1
    else:
        raise ValueError(f"Unsupported dtype: {out_dtype}")
    
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.dtype == torch.float8_e4m3fn, f"A must be e4m3fn, found {A.dtype}"
    assert B.dtype == torch.float8_e4m3fn, f"B must be e4m3fn, found {B.dtype}"
    assert A.ndim == 2, f"A must be 2D, found A.ndim={A.ndim}"
    assert B.ndim == 2, f"B must be 2D, found B.ndim={B.ndim}"
    
    configs = KERNELCFG
    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    BLOCK_K = configs["BLOCK_SIZE_K"]

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[0]
    assert A.shape[1] == B.shape[1], f"Incompatible inner dimensions: {A.shape[1]} vs {B.shape[1]}"
    assert M % BLOCK_M == 0, f"M dim of A must be multiple of BLOCK_SIZE_M: found {M} % {BLOCK_M}"
    assert N % BLOCK_N == 0, f"N dim of B must be multiple of BLOCK_SIZE_N: found {N} % {BLOCK_N}"
    assert K % BLOCK_K == 0, f"K dim must be multiple of BLOCK_SIZE_K: found {K} % {BLOCK_K}"

    assert scaleA.is_contiguous(), "A must be contiguous"
    assert scaleB.is_contiguous(), "B must be contiguous"
    assert scaleA.ndim == 4, f"scaleA must be 4D, found scaleA.ndim={scaleA.ndim}"
    assert scaleB.ndim == 4, f"scaleB must be 4D, found scaleB.ndim={scaleB.ndim}"
    assert scaleA.dtype == torch.uint8, f"scaleA must be uint8 (e8m0), found {scaleA.dtype}"
    assert scaleB.dtype == torch.uint8, f"scaleB must be uint8 (e8m0), found {scaleB.dtype}"
    assert scaleA.shape[2] == 128 and scaleA.shape[3] == 4, f"scaleA must be tiled in 128x4, found scaleA.shape={scaleA.shape}"
    assert scaleB.shape[2] == 128 and scaleB.shape[3] == 4, f"scaleB must be tiled in 128x4, found scaleB.shape={scaleB.shape}"


    rep_m, rep_n, rep_k = 1, 2, 1
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])

    tiled_scaleA = scaleA.reshape((1,) + scaleA.shape[:2] + (2, 256))
    tiled_scaleB = scaleB.reshape((1,) + scaleB.shape[:2] + (2, 256))
    a_scale_desc = TensorDescriptor.from_tensor(tiled_scaleA, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(tiled_scaleB, block_shape=b_scale_block_shape)

    output = torch.empty((M, N), dtype=out_dtype, device="cuda")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc, a_scale_desc,
        b_desc, b_scale_desc,
        c_desc,
        M, N, K, dtype_dst_id,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m, rep_n, rep_k,
        configs["num_stages"],
    )
    return output


if __name__ == "__main__":
    # M, N, K = 128, 256, 512
    # M, N, K = 128, 64, 1088
    # M, N, K = 8192, 4096, 1024
    M, N, K = 512, 256, 768
    A = torch.rand((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.rand((N, K), dtype=torch.bfloat16, device="cuda")

    C_ref = torch.matmul(A, B.T)

    Aq, scaleA = quantize_rowwise(A)
    Bq, scaleB = quantize_rowwise(B)

    C = mxfp8_matmul_NT(Aq, scaleA, Bq, scaleB, torch.bfloat16)

    torch.testing.assert_close(C_ref, C, atol=0.0, rtol=5e-2)
    print("pass.")
    print("joto")
 