## FP8
```diff
--- fp16_train_mnist.py 2025-07-17 21:23:28.260334444 +0000
+++ fp8_train_mnist.py  2025-07-17 23:17:04.745249874 +0000
@@ -6,7 +6,8 @@
 from tqdm import tqdm
 
 from models.vit import TinyViT
-from torch.amp import autocast, GradScaler
+import transformer_engine.pytorch as te
+from transformer_engine.common import recipe
 
 # ── 1. Hyper-params ────────────────────────────────────────────────────────────
 BATCH_SIZE   = 64
@@ -32,7 +33,7 @@
 print(model)
 optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 criterion = nn.CrossEntropyLoss()
-scaler = GradScaler()
+fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)
 # ── 4. Training loop ───────────────────────────────────────────────────────────
 for epoch in range(1, EPOCHS + 1):
     model.train()
@@ -41,14 +42,13 @@
     for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
         x, y = x.to(DEVICE), y.to(DEVICE)
 
-        with autocast(dtype=torch.float16, device_type=DEVICE): # Casts operations to mixed precision
+        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) as fp8_ctx:
             logits = model(x)
             loss   = criterion(logits, y)
 
         optimizer.zero_grad()
-        scaler.scale(loss).backward() # Scale the loss before backward pass
-        scaler.step(optimizer)        # Unscale gradients and step optimizer
-        scaler.update()               # Update the scale factor for next iteration
+        loss.backward() # Scale the loss before backward pass
+        optimizer.step()
 
         loss_sum += loss.item() * y.size(0)
         preds = logits.argmax(1)
```
recipe

What is DelayScaling?
* single loss scaling factor doesn't work in FP8. Why?
* need respective scaling factor for each tensor and need to be dynamic. one option is to do it just in time but calculating scaler require collection of few datapoints, aggregrate stats, overhead may trumph the gain of fp8 compute downstream. Delay Scaling solves this issue by keeping tab of the most recent stats of a tensor and dynamically resolute to a scaler value.

why gradient scaler is not required any more? i think this is orthogonal, can still use it.

how to know it is actually using FP8?
we wont observe fp8 tensor because it only happens internally (autocast and decast) for mma.
i do observe some speed up, also not much because it is too small to saturate the compute
function-based context manager, how to access it though?
good to see it, by get to the scaler of each tensor
good to see the history buffer

find out what enabled in autocast do?

 backward call needs to happen outside of the fp8_autocast context manager.

MXFP8
inner dimension
e8m0 unsigned
so MXFP8 is just-in-time?