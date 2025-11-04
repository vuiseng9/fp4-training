import torch
import math

def swizzle_rowwise_scale(scale_mat):
    tile_dim_y = 128
    tile_dim_x = 4

    swiz_tile_dim_y = 32
    swiz_tile_dim_x = 16

    orig_y, orig_x = scale_mat.shape

    ntile_y = math.ceil(orig_y / tile_dim_y)
    ntile_x = math.ceil(orig_x / tile_dim_x)

    padded_shape = (ntile_y * tile_dim_y, ntile_x * tile_dim_x)
    padded_scale_mat = torch.zeros(padded_shape, dtype=scale_mat.dtype, device=scale_mat.device)
    padded_scale_mat[:orig_y, :orig_x] = scale_mat
    padded_scale_mat = padded_scale_mat.view(ntile_y, tile_dim_y, ntile_x, tile_dim_x).permute(0,2,1,3)

    swizzle_scales = torch.zeros((ntile_y*swiz_tile_dim_y, ntile_x*swiz_tile_dim_x), dtype=padded_scale_mat.dtype, device=padded_scale_mat.device).contiguous()

    for ty in range(ntile_y):
        for tx in range(ntile_x):
            scale_tile = padded_scale_mat[ty, tx]
            chunked_scale_tile = scale_tile.view(4, 32, 4)

            for i in range(4):
                swizzle_scales[
                    ty*swiz_tile_dim_y:(ty+1)*swiz_tile_dim_y,
                    (tx*swiz_tile_dim_x)+(i*4):(tx*swiz_tile_dim_x)+(i*4)+4
                ] = chunked_scale_tile[i]

    return swizzle_scales


def swizzle_colwise_scale(scale_mat):
    tile_dim_y = 4
    tile_dim_x = 128

    swiz_tile_dim_y = 32
    swiz_tile_dim_x = 16

    orig_y, orig_x = scale_mat.shape

    ntile_y = math.ceil(orig_y / tile_dim_y)
    ntile_x = math.ceil(orig_x / tile_dim_x)

    padded_shape = (ntile_y * tile_dim_y, ntile_x * tile_dim_x)
    padded_scale_mat = torch.zeros(padded_shape, dtype=scale_mat.dtype, device=scale_mat.device)
    padded_scale_mat[:orig_y, :orig_x] = scale_mat
    padded_scale_mat = padded_scale_mat.view(ntile_y, tile_dim_y, ntile_x, tile_dim_x).permute(0,2,1,3)

    swizzle_scales = torch.zeros((ntile_y*swiz_tile_dim_y, ntile_x*swiz_tile_dim_x), dtype=padded_scale_mat.dtype, device=padded_scale_mat.device).contiguous()

    for ty in range(ntile_y):
        for tx in range(ntile_x):
            scale_tile = padded_scale_mat[ty, tx]
            chunked_transposed = scale_tile.view(4, -1, 32).permute(1, 0, 2).permute(0,2,1)
            swizzle_logical = chunked_transposed.permute(1,0,2).reshape(swiz_tile_dim_y, swiz_tile_dim_x)
            
            swizzle_scales[
                ty*swiz_tile_dim_y:(ty+1)*swiz_tile_dim_y,
                tx*swiz_tile_dim_x:(tx+1)*swiz_tile_dim_x
                ] = swizzle_logical
    return swizzle_scales



if __name__ == "__main__":
    nrow=128*1
    ncol=4*1

    scale_mat = torch.arange(nrow*ncol, dtype=torch.int16).view(nrow, ncol)
    swmat = swizzle_rowwise_scale(scale_mat)

    nrow=4*1
    ncol=128*1

    scale_mat = torch.arange(nrow*ncol, dtype=torch.int16).view(nrow, ncol)
    swmat = swizzle_colwise_scale(scale_mat)

    # inner (K for A and B, and M for C or D) and outer (M for A, and N for B, C and D)
    # A[M, K], outer=M, inner=K 
    # B[K, N], outer=N, inner=K
    # C[M, N], outer=N, inner=M
    # D[M, N], outer=N, inner=M
    # why M is the inner for D? D[M, K] will be input to next layer which will be a B[K, N]. Since K is inner, M of D should be inner?

    from collections import OrderedDict, defaultdict

    def raw_to_offset(outer, inner):
        return (outer % 32) * 16 + (outer // 32) * 4 + inner

    # use the following to understanding how 128x4 is swizzled to 32x16
    # --------------------------------------------------------------------------------
    # A[M, K], outer=M, inner=K
    # scaleA[M, K/32]
    offset_to_scale_a = OrderedDict()

    for r in range(128):
        for c in range(4):
            offset = raw_to_offset(r, c)
            offset_to_scale_a[offset] = (r, c)
            # print(f"scale_A[{r:3},{c:3}] = {offset}")

    # post swizzled
    for linear_id in sorted(offset_to_scale_a.keys()):
        print(f"offset {linear_id:3} -> scaleA{offset_to_scale_a[linear_id]}")              

    print("-"*100)
    # use the following to understanding how 4x128 is swizzled to 32x16
    # --------------------------------------------------------------------------------
    # B[K, N], outer=N, inner=K
    # scaleB[K/32, N]
    offset_to_scale_b = OrderedDict()

    for r in range(4): # K inner
        for c in range(128): # N outer
            offset = raw_to_offset(outer=c, inner=r)
            offset_to_scale_b[offset] = (r, c)

    # post swizzled
    for linear_id in sorted(offset_to_scale_b.keys()):
        print(f"offset {linear_id:3} -> scaleB{offset_to_scale_b[linear_id]}")  

    print("end.")
