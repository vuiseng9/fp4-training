import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from custom import CublasltNvfp4Linear

def noraise_allclose(*args, **kwargs):
    try:
        # Attempt the assertion
        torch.testing.assert_close(*args, **kwargs)
        print("✅ Tensors are close!")
    except AssertionError as e:
        # If it fails, catch the error and print its message
        print("❌ Tensors are NOT close. See comparison below:")
        print(e)

# repi = recipe.MXFP8BlockScaling()
repi = recipe.NVFP4FwdMXFP8BwdScaling()

IC=128
OC=64
# layer = CublasltNvfp4Linear(in_features=IC, out_features=OC, bias=False)
layer = te.Linear(in_features=IC, out_features=OC, bias=True).to("cuda")

ref_layer = CublasltNvfp4Linear(in_features=IC, out_features=OC, bias=True).to("cuda")

sd = {}
sd['weight'] = layer.state_dict()['weight']
sd['bias'] = layer.state_dict()['bias']
ref_layer.load_state_dict(sd)

x = torch.randn(64, 17, IC).to("cuda")
y = torch.randn(64, 17, OC).to("cuda")

criterion = nn.MSELoss()

layer.train()

with te.fp8_autocast(fp8_recipe=repi) as fp8_ctx:
    logits = layer(x)
    loss   = criterion(logits, y)

ref_y = ref_layer(x)

noraise_allclose(logits, ref_y, atol=0.10, rtol=0.00)

print("forward done.", flush=True)
# loss.backward() # Scale the loss before backward pass
print("end.", flush=True)
