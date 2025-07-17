## FP16 Training
```diff
--- fp32_train_mnist.py 2025-07-14 22:21:49.727396970 -0700
+++ fp16_train_mnist.py 2025-07-14 22:41:38.454304963 -0700
...
+from torch.amp import autocast, GradScaler
...

 optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 criterion = nn.CrossEntropyLoss()

+scaler = GradScaler()
 # ── 4. Training loop ───────────────────────────────────────────────────────────
 for epoch in range(1, EPOCHS + 1):
     model.train()

     for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
         x, y = x.to(DEVICE), y.to(DEVICE)
 
-        logits = model(x)
-        loss   = criterion(logits, y)
+        with autocast(dtype=torch.float16, device_type=DEVICE): # Casts operations to mixed precision
+            logits = model(x)
+            loss   = criterion(logits, y)
 
         optimizer.zero_grad()
-        loss.backward()
-        optimizer.step()
+        scaler.scale(loss).backward() # Scale the loss before backward pass
+        scaler.step(optimizer)        # Unscale gradients and step optimizer
+        scaler.update()               # Update the scale factor for next iteration
 
         loss_sum += loss.item() * y.size(0)
         preds = logits.argmax(1)
```
1. Above is the standard FP16 training loop with native PyTorch. PyTorch has integrated Nvidia APEX, specifically O1 stage. What is the O1?
    * Ops are meant to execute on FP16 on hardware, there is a "safe" list of ops. With the context manager, PyTorch autocast the ops according to the whitelist. See ops that can be cast to FP16 [here](https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16).
    * What happens to the weight, gradient? Weights are in kept in FP32. The casting occurs dynamically, XW matmul will have thier respective inputs autocasted and fp16 mma and uncasted if needed by downstream op. This happens to backward computation as well. Since grad is a result of these fp16 ops, they are kept at fp16 and only upcasted during weight update. In short, forward and backward computation in fp16, weight in FP32, gradient FP16.
    * how about optimizer states, e.g. momentum and variance in adam, they are also kept in FP32. Therefore, quoting Huggingface's documentation, "fp16 isn’t memory-optimized because the gradients that are computed in fp16 are converted back to fp32 during the optimization step. You may end up using more GPU memory, especially for small batch sizes, because there are now two versions (fp16 and fp32) of the model on the GPU."
    * ZeRO paper hinges on the 4-byte weight, 2-byte gradient, 4-byte momentum and 4-byte variance, 14 bytes per parameter footprint for the needs of memory optimization.
2. Why gradient scaler is needed? is it related to the casting operation?
    * because FP16 has narrower range and precision (BF16 has same range to FP32), scaler is to avoid overflow and underflow. Nothing much related to casting at this point, mainly to move the gradient distribution to save value range for computation. How scaler can be automatic? it start with large value and dynamically check if inf/nan encountered and tweak the scaler value. Details: scale by power of 2 so mathematically lossless during unscale.





delayed scaling. This strategy chooses the scaling factor based on the maximums of absolute values seen in some number of previous iterations. This enables full performance of FP8 computation, but requires storing the history of maximums as additional parameters of the FP8 operators.

In FP8, each tensor has a single FP32 scaling factor, so all values in the tensor need to “fit” within the dynamic range of the FP8 datatype.  This requires using the less precise E5M2 format to represent some tensors in the network (like gradients).


how grad scaler work under the hood? scale the loss only? based on what value, all linear
the backward pass
aggregation part


Weights → per-output-channel, block-of-32 in-features