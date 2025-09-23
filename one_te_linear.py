import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# repi = recipe.MXFP8BlockScaling()
repi = recipe.NVFP4BlockScaling()

IC=128
OC=64
layer = te.Linear(in_features=IC, out_features=OC, bias=False)

layer.to("cuda")

x = torch.randn(64, 17, IC).to("cuda")
y = torch.randn(64, 17, OC).to("cuda")

criterion = nn.MSELoss()

layer.train()
with te.fp8_autocast(fp8_recipe=repi) as fp8_ctx:
    logits = layer(x)
    loss   = criterion(logits, y)

print("forward done.", flush=True)
loss.backward() # Scale the loss before backward pass
print("end.", flush=True)


# X[64, 17, 128]
# W[64, 128] oc, ic

# dY[1088, 64] n, oc

# A
# X[1088, 128] n, ic
# colwise
# X[544, 128] n, ic
# cublas N: X.T[128, 544]*

# B
# dY[1088, 64] n, oc
# colwise
# dY[544, 64] n, oc
# cublas T: dY.T[64, 544] -> dY[544, 64]*

# mxfp8
# [2025-09-10 17:32:51][cublasLt][133267][Api][cublasLtMatmulAlgoGetHeuristic] 
# Adesc=[type=R_8F_E4M3 rows=128 cols=1088 ld=128] 
# Bdesc=[type=R_8F_E4M3 rows=64 cols=1088 ld=64] 
# Cdesc=[type=R_32F rows=128 cols=64 ld=128] 
# Ddesc=[type=R_32F rows=128 cols=64 ld=128] 
# preference=[maxWavesCount=0.0 maxWorkspaceSizeinBytes=33554432] 
# computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transb=OP_T 
#              aScalePointer=0x78a0ed1fba00 bScalePointer=0x78a0ed1fcc00 
#              aScaleMode=VEC32_UE8M0 bScaleMode=VEC32_UE8M0]

# nvfp4
# [2025-09-10 17:36:09][cublasLt][138099][Api][cublasLtMatmulAlgoGetHeuristic] 
# Adesc=[type=R_4F_E2M1 rows=128 cols=64 ld=128] 
# Bdesc=[type=R_4F_E2M1 rows=128 cols=1088 ld=128] 
# Cdesc=[type=R_32F rows=64 cols=1088 ld=64] 
# Ddesc=[type=R_32F rows=64 cols=1088 ld=64] 
# preference=[maxWavesCount=0.0 maxWorkspaceSizeinBytes=33554432] 
# computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T 
#               aScalePointer=0x72d1c513fa00 bScalePointer=0x72d1c513fe00 
#               aScaleMode=VEC16_UE4M3 bScaleMode=VEC16_UE4M3]

# wgrad failed
# [2025-09-10 17:36:09][cublasLt][138325][Api][cublasLtMatmulAlgoGetHeuristic] 
# Adesc=[type=R_4F_E2M1 rows=128 cols=1088 ld=128] 
# Bdesc=[type=R_4F_E2M1 rows=64 cols=1088 ld=64] 
# Cdesc=[type=R_32F rows=128 cols=64 ld=128] 
# Ddesc=[type=R_32F rows=128 cols=64 ld=128] 
# preference=[maxWavesCount=0.0 maxWorkspaceSizeinBytes=33554432] 
# computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transb=OP_T 
#              aScalePointer=0x72d1c51d1c00 bScalePointer=0x72d1c51d3e00 
#              aScaleMode=VEC16_UE4M3 bScaleMode=VEC16_UE4M3]



# [Lt] Algo candidates by types: 2
# [Lt] id=  70 | init=success        | checkRet=an unsupported value or parameter was passed to the function | result.state=an unsupported value or parameter was passed to the function | ws=0
# [init-default] tile=0 stages=0 splitK=1 red=0 cta=0 inner=0 cluster=0 custom=0
# [Lt] id=  71 | init=success        | checkRet=an unsupported value or parameter was passed to the function | result.state=an unsupported value or parameter was passed to the function | ws=0
# [init-default] tile=0 stages=0 splitK=1 red=0 cta=0 inner=0 cluster=0 custom=0
# [heuristic] id=  70 | result.state=success        | ws=0
# [heuristic] tile=20 stages=37 splitK=1 red=0 cta=0 inner=0 cluster=6 custom=0
# name=NVIDIA B200 cc=10.0
# cuBLAS version=120901
# heuristic_returned=1  first_ws=0
# heuristic picked algoId=70
# compute=68 scale=0 transA=1 transB=0 epilogue=1 ptrMode=0
# A_scale_mode=1 ptr=0x7e123793fa00 | B_scale_mode=1 ptr=0x7e123793fe00 | C_scale_mode=0 ptr=(nil) | D_scale_mode=0 ptr=(nil)
# A: type=33 order=0 rows=128 cols=64 ld=128 batch=1 stride=0
# B: type=33 order=0 rows=128 cols=1088 ld=128 batch=1 stride=0
# C: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
# D: type=0 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
# forward done.

# [New Thread 0x7e122d9ff6c0 (LWP 634220)]
# [Lt] Algo candidates by types: 2
# [Lt] id=  70 | init=success        | checkRet=an unsupported value or parameter was passed to the function | result.state=an unsupported value or parameter was passed to the function | ws=0
# [init-default] tile=0 stages=0 splitK=1 red=0 cta=0 inner=0 cluster=0 custom=0
# [Lt] id=  71 | init=success        | checkRet=an unsupported value or parameter was passed to the function | result.state=an unsupported value or parameter was passed to the function | ws=0
# [init-default] tile=0 stages=0 splitK=1 red=0 cta=0 inner=0 cluster=0 custom=0
# [heuristic] id=  70 | result.state=-              | ws=0
# [heuristic] tile=0 stages=0 splitK=1 red=0 cta=0 inner=0 cluster=0 custom=0
# name=NVIDIA B200 cc=10.0
# cuBLAS version=120901
# heuristic_returned=0  first_ws=0
# heuristic picked algoId=70
# compute=68 scale=0 transA=0 transB=1 epilogue=1 ptrMode=0
# A_scale_mode=1 ptr=0x7e12379d1c00 | B_scale_mode=1 ptr=0x7e12379d3e00 | C_scale_mode=0 ptr=(nil) | D_scale_mode=0 ptr=(nil)
# A: type=33 order=0 rows=128 cols=1088 ld=128 batch=1 stride=0
# B: type=33 order=0 rows=64 cols=1088 ld=64 batch=1 stride=0
# C: type=0 order=0 rows=128 cols=64 ld=128 batch=1 stride=0
# D: type=0 order=0 rows=128 cols=64 ld=128 batch=1 stride=0