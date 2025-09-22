import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from custom import CublasltNvfp4Linear
# repi = recipe.MXFP8BlockScaling()
# repi = recipe.NVFP4BlockScaling()

IC=128
OC=64
layer = CublasltNvfp4Linear(in_features=IC, out_features=OC, bias=False)

layer.to("cuda")

x = torch.randn(64, 17, IC).to("cuda")
y = torch.randn(64, 17, OC).to("cuda")

criterion = nn.MSELoss()

layer.train()

logits = layer(x)
loss   = criterion(logits, y)

print("forward done.", flush=True)
loss.backward() # Scale the loss before backward pass
print("end.", flush=True)
