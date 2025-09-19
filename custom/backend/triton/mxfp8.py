import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from custom.quantize import _q_mxfp8_rowwise, _q_mxfp8_colwise
from functools import partial

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

def tile_matrix(tensor, ty, tx):
    # pad to tile boundary, return 5D tensor 
    if tensor.shape[0] % ty != 0 or tensor.shape[1] % tx != 0:
        padded = torch.zeros((ceilto(tensor.shape[0], ty), ceilto(tensor.shape[1], tx)), dtype=tensor.dtype, device=tensor.device)
        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
        tensor = padded
    
    tiled = tensor.view(tensor.shape[0]//ty, ty, tensor.shape[1]//tx, tx).permute(0,2,1,3).unsqueeze(dim=0).contiguous()
    return tiled

tile_128x4 = partial(tile_matrix, ty=128, tx=4)
tile_4x128 = partial(tile_matrix, ty=4, tx=128)

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
        output_dtype = tl.float16
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


def mxfp8_matmul_NT(A, B, dtype_dst):
    # A: (M, K) FP32
    # B: (N, K) FP32
    if dtype_dst == torch.float32:
        dtype_dst_id = 0
    elif dtype_dst == torch.float16:
        dtype_dst_id = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst_id = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")
    
    rep_m, rep_n, rep_k = 1, 2, 1
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

    configs = KERNELCFG
    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    BLOCK_K = configs["BLOCK_SIZE_K"]
    VEC_SIZE = configs["VEC_SIZE"]

    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[0]
    assert A.shape[1] == B.shape[1], "Incompatible inner dimensions"
    assert M % BLOCK_M == 0, "M must be multiple of BLOCK_SIZE_M"
    assert N % BLOCK_N == 0, "N must be multiple of BLOCK_SIZE_N"
    assert K % BLOCK_K == 0, "K must be multiple of BLOCK_SIZE_K"

    Aq, scaleA = quantize_rowwise(A)
    Bq, scaleB = quantize_rowwise(B)

    a_desc = TensorDescriptor.from_tensor(Aq, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(Bq, [BLOCK_N, BLOCK_K])

    tiled_scaleA = tile_128x4(scaleA)
    tiled_scaleA = tiled_scaleA.reshape(tiled_scaleA.shape[:3] + (2, 256))

    tiled_scaleB = tile_128x4(scaleB)
    tiled_scaleB = tiled_scaleB.reshape(tiled_scaleB.shape[:3] + (2, 256))

    a_scale_desc = TensorDescriptor.from_tensor(tiled_scaleA, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(tiled_scaleB, block_shape=b_scale_block_shape)

    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst_id,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output


if __name__ == "__main__":
    M, N, K = 8192, 4096, 1024
    A = torch.rand((M, K), dtype=torch.float16, device="cuda")
    B = torch.rand((N, K), dtype=torch.float16, device="cuda")

    C_ref = torch.matmul(A, B.T)
    C = mxfp8_matmul_NT(A, B, torch.float16)

    torch.testing.assert_close(C_ref, C.to(torch.float16), atol=0.0, rtol=5e-2)
    print("pass.")