## Quantized Training in FP8/FP4
*Concepts and Reference Pytorch Implementation using [cuBLASLt][doc_cublaslt] and [Microxcaling][ghmsmx].*

Narrow-precision training is rapidly becoming mainstream. This repo offers a concise technical walkthrough and reference implementation targeting modern hardware (e.g., Blackwell B200). The goal is to help practitioners understand and customize low-precision layer end-to-end, not just run a black-box recipe.

Jump to:
- [Hit the Ground Running](#hit-the-ground-running-🚀)
- [Low Precision Training Outcomes on TinyViT/MNIST](#training-outcomes)
- [Coding Guide on using cuBLASlt and Microxcaling](#coding-guide)
- [The Three GEMMs of Training](#the-three-gemms-of-training)
- [1D Block Quantization, Microscaling (MX) Format and NVFP4](#1d-block-quantization-mx-format-and-nvfp4)
- [Varying Axis of Quantization](#varying-axis-of-quantization)
- [Recent Trends in FP4 Training Research](#recent-trends-in-fp4-training-research)
- [Future Plan](#future-plan)
- [Further Reading and References](#references)

---
### The Three GEMMs of Training
*a.k.a. the trilogy behind FP4/FP8 speedups*

The premise of low precision training is the "*Speedups*" by mapping the heavy math in fewer bit representation where corresponding hardware runs faster. On FP4-supported HW, e.g. [NVIDIA Blackwell (B200)](https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703) and [AMD CDNA 4 (MI350X)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-platform-brochure.pdf), **FP4** matmul peak throughput is about **2× of FP8**, **4× over FP/BF16**.

Most modern models are Transformers. The main workhorses are linear projections and attentions which are fundamentally matrix multiplications (matmuls). If we can execute these matmuls on the new FP4/FP8 units, we get speedups in training (inference too). In linear algebra libraries these matmuls are often referred to as GEMMs, short for GEneral Matrix-Matrix Multiplications.

However, not all operations in a Transformer can be safely pushed to low precision. Numerical stability and training convergence constrain  optimizers, normalization, softmax, and other sensitive components to remain in higher precision. As a result, most research and practical systems focus on linear layers first, as they dominate training compute and tend to better "absorb" quantization effects.

We now walk through the three matmuls that form the *"trilogy"* of linear-layer training, and how quantization applies to each. A diagram follows to illustrate the discussion. 
![](./assets/low-fp-gemm.png)
* **MatMul 1** for computing forward pass of linear layer:
   
   &emsp; $Y = X W^{T}$ &emsp; where input $X$ is $(N, IC)$, weights $W$ is $(OC, IC)$ following `torch` layout, and output $Y$ is $(N, OC)$. For brevity, Transformer's batch size and sequence length are collapsed into $N$.

* **MatMul 2** in the backward pass for computing **gradient w.r.t. inputs** :

   &emsp; $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\frac{\partial Y}{\partial X} =GW$ &emsp; where $G = \frac{\partial L}{\partial Y}$, is $(N, OC)$, the backprop incoming gradient. $\frac{\partial L}{\partial X}$ has shape of $(N, IC)$ 

* **MatMul 3** in the backward pass for computing **gradient w.r.t. weights**:

   &emsp; $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y}\frac{\partial Y}{\partial W} =G^{T}X$ &emsp; where $\frac{\partial L}{\partial W}$ has the same shape as $W$, $(OC, IC)$


Essentially, $X, W, G$ must be quantized to target precision (FP8/4) before we feed them to the matrix engines. Notice the quantization operators in diagram above. The quantization used in training today generally follows the form below (technically  known as symmetric quantization). Given a matrix $M$, quantization produces a quantized matrix $Q_M$ and a scale $s_M$:

&emsp; $Q_M = \mathrm{round}(M / s_M)$ &emsp; where &ensp; $s_M = \frac{\max(|M|)}{q_{\max}}$

Here, $q_{\max}$ denotes the maximum representable magnitude in the target precision. $s_M$ is a *scalar* scaler 🎯😁.

Now consider a matrix multiplication $A@B$, we quantize $A$ and $B$ into $Q_{A}, Q_{B}$ with scales $s_{A}, s_{B}$ respectively. The low-precision matmul becomes:

&emsp; $A@B \approx s_A \cdot Q_A @ s_B \cdot Q_B = (s_A \cdot s_B)(Q_A @ Q_B)$

The execution above is hardware-accelerated, output of low-precision matmul $Q_{A} @ Q_{B}$ results will be mapped (*dequantized*) to the original high-precision using $s_A, s_B$.

### 1D Block Quantization, MX Format and NVFP4

Quantization introduces distortion, which can cause training divergence if not properly managed. A key lever for minimizing distortion is granularity. Granularity refers to how to group the elements within a matrix such that each group is quantized independently with its own scale. Smaller groups tends to bound the dynamic range, which effectively increases representable precision in low-bit formats, thereby reducing quantization error. In principle, grouping size can take arbitrary shapes. A matrix can be quantized with:
* One scale for the whole matrix (per-tensor)
* One scale per row or per column
* One scale per block (block / group quantization), e.g. 4×4, 8×32, 1×16, etc.

A frontier example is [DeepSeek-V3][dsv3], which trains in FP8 using 128×128 weight blocks and 1×128 activation blocks, a configuration that is friendly to Hopper architecture and helps mitigate the sensitivity to outliers in per-tensor quantization.

Pushing narrower precision demands finer granularity. Varying choices among hardware vendor would be a nightmare for model portability and interoperability. **Microscaling Formats (MX)**, a specification from the Open Compute Project (OCP), aims to prevent such fragmentation by establishing a common low-precision representation for vendors and model providers. At its core, **MX defines** a 1D block size of 32 elements, along with the encoding format for the scale (8-bit exponent) and quantized values (FP4/FP6/FP8, including ExMy, NaN/Inf/subnormal). MXFP4/6/8 denote MX-compliant formats.** See the [OCP MX spec][ocp_mx] for details.

As of Q3/Q4 2025, on top of MXFP8/6/4, NVIDIA Blackwell also supports [NVFP4][blog_nvfp4_i]. The key differences are that **NVFP4** uses 16-element blocks instead of 32 and employs an FP8 scale instead of an 8-bit exponent**. We will experiment with MXFP8 and NVFP4 in our cuBLASLt-based implementation later.

| Format   | Block Size | Scale Type | Value Type           |
|----------|:----------:|:----------:|----------------------|
| MXFP8    | 32         | E8M0       | FP8: E5M2 / E4M3     |
| MXFP6    | 32         | E8M0       | FP6: E3M2 / E2M3     |
| MXFP4    | 32         | E8M0       | FP4: E2M1            |
| NVFP4    | 16         | FP8 (E4M3) | FP4: E2M1            |

> ExMy has a leading sign bit except E8M0. MXINT8 is also defined in the MX spec.

### Varying Axis of Quantization

"1D" block quantization means the blocks are taken along **one** matrix axis, even though the physical grouping is still 2D. For example, MX groups contiguous 32 elements along a row or a column (NVFP4 uses 16 elements), so the block shape is effectively 1×K or K×1 with K is the block size.

This raises a key question: along which axis should we quantize? **Along the contraction (inner) axis of the matmul.**

If you inspect the three training GEMMs closely, each $W, X, G$ must be quantized along different axes depending on the matmul. As a result, the same tensor requires both axes of quantization. In the diagram, we normalize everything to row-wise quantization and insert transposes to match our equations. In practice, implementations choose how to handle this. For example:
1. [Transformer Engine][te] keeps both row-wise and column-wise quantized copies to avoid transposes at runtime.
2. Some work uses double quantization (as shown on the right of the diagram), i.e., re-quantizing an already (de)quantized tensor along the other axis. This avoids storing two copies, at the cost of additional quantization error.

That's it! These are the key concepts behind low-precision training on state-of-the-art hardware today. Next, we will implement a custom PyTorch Linear module that performs the trio of GEMMs using cuBLASLt with official MX quantization.

---
### Hit the Ground Running 🚀

Setup: Use the prebuilt Docker image on a B200 GPU. Other Blackwell cards (e.g., RTX 50-series and PRO 6000) are not supported, we use for comparison are not enabled on them yet.
```
docker run ... <to be added soon>
```
If you'd like to customize, refer `docker/Dockerfile`. Note: building Transformer Engine from source is non-trivial, if you do, start from an NVIDIA Docker image. However, if you only want to build and run our implementation, native PyTorch + CUDA Toolkit is sufficient (no TE required).

**What to run?**
We provide two scripts that demonstrate quantized training via different `Linear` implementations:
1. Our custom MXFP8 / NVFP4 path via [cuBLASLt][doc_cublaslt] + [Microxcaling][ghmxfork]
2. Nvidia's [Transformer Engine][te] recipe for comparison

Both scripts train a tiny ViT (single Transformer block) on MNIST. See [Coding Guide](#coding-guide) for walkthrough of our implementation and training quality comparison right after this section.

```bash
# (1) Our custom cuBLASLt + Microxcaling backend
python main_train_tinyvit_mnist.py 
   --impl torch             # out-of-the-box pytorch baseline 
   --impl custom_py         # inherited linear to illustrate the 3 gemms and template for custom kernel
   --impl custom_aten       # custom cpp-cuda backend using aten addmm/mm  
   --impl cublaslt          # custom cpp-cuda backend using cublaslt for bf16 / fp32 matmul
   --impl cublaslt_mxfp8    # mxfp8 via microxcaling + cublaslt mxfp8 matmul
   --impl cublaslt_nvfp4    # nvfp4 via microxcaling + cublaslt nvfp4 matmul
   --impl cublaslt_nvf4_fw_mxf8_bw # nvfp4 forward + mxfp8 backward
   -h # see options, default: batch size 64, 3 epoch, lr 1e-3

# Notes:
# All of the above are using bf16 as main compute 
#  via torch.autocast as the standard baseline precision today.
# i.e. Input and weight of linear are dynamically cast to bf16 in forward pass, 
#  the quantization is from bf16 to the target low precision.
# We also support fp32, just add --fp32
```

```bash
# (2) Transformer Engine (TE)
python main_te_train_tinyvit_mnist.py 
   --recipe base      # TE Linear layer, trainable with torch.autocast
   --recipe fp8       # Float8CurrentScaling recipe (per-tensor fp8)
   --recipe mxfp8     # MXFP8BlockScaling recipe
   --recipe nvfp4     # NVFP4BlockScaling recipe
   -h # see options, default: batch size 64, 3 epoch, lr 1e-3
   # similarly just add --fp32 for main compute in full precision
```

**Verifying matmul precision through cuBLASLT logging:**
```bash
export CUBLASLT_LOG_LEVEL=2
export CUBLASLT_LOG_FILE=./log.cublaslt # optional. If not set, 
# cublaslt logs to stdout, harder to see training progress.
```
Example log lines: 
* `A/Bdesc` reports details of input matrices A, B (`R_8F_E4M3`, `R_4F_E2M1`), `Ddesc` for matmul output (`R_16BF`, `R_32F`).
* Layout transpose: `transa=OP_T` or `transb=OP_T`, no report means no transpose.
* MX or NV blocking? `aScaleMode=VEC32_UE8M0 bScaleMode=VEC32_UE8M0` vs  `aScaleMode=VEC16_UE4M3 bScaleMode=VEC16_UE4M3`
* `computeType=COMPUTE_32F` is expected for mxfp8/nvfp4 matmuls as they are accumulated to f32 internally.
```bash
# mxfp8
[2025-11-03 19:02:38][cublasLt][557785][Trace][cublasLtMatmul] A=0X736CB315E800 
Adesc=[type=R_8F_E4M3 rows=64 cols=1088 ld=64] B=0X736C9B644000 
Bdesc=[type=R_8F_E4M3 rows=128 cols=1088 ld=128] C=0X0 
Cdesc=[type=R_16BF rows=64 cols=128 ld=64] D=0X736CB30A4400 
Ddesc=[type=R_16BF rows=64 cols=128 ld=64] 
computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transb=OP_T 
aScalePointer=0x736cb30b4c00 bScalePointer=0x736cb30ba200 
aScaleMode=VEC32_UE8M0 bScaleMode=VEC32_UE8M0] 
algo=[algoId=66 tile=MATMUL_TILE_128x128 
stages=MATMUL_STAGES_128xAUTO customOption=3 clusterShape=CLUSTER_SHAPE_1x1x1] 
workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=1 stream=0X0

# nvfp4
[2025-11-03 19:07:51][cublasLt][570888][Trace][cublasLtMatmul] A=0X75056B1F7000 
Adesc=[type=R_4F_E2M1 rows=1088 cols=64 ld=1088] B=0X75051DFCDE00 
Bdesc=[type=R_4F_E2M1 rows=1088 cols=64 ld=1088] C=0X0 
Cdesc=[type=R_32F rows=64 cols=64 ld=64] D=0X75051D9F8A00 
Ddesc=[type=R_32F rows=64 cols=64 ld=64] 
computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T 
aScalePointer=0x75051d9da400 bScalePointer=0x75051dfbaa00 
aScaleMode=VEC16_UE4M3 bScaleMode=VEC16_UE4M3] 
algo=[algoId=70 tile=MATMUL_TILE_128x128 
stages=MATMUL_STAGES_256xAUTO clusterShape=CLUSTER_SHAPE_2x1x1 schedulingMode=0] 
workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=1 stream=0X0
```

---
### Training Outcomes on TinyViT/MNIST

*coming soon*

---
### Coding Guide

To experiment with FP8/FP4 training, the main component we customize is the Linear layer. This requires three pieces to work together:
(1) cuBLASLt, to drive the hardware-accelerated low-precision GEMMs,
(2) quantization, where we rely on the official Microxcaling library (forked for customization), and
(3) a minimal model + training loop to validate correctness and training quality. We use TinyViT on MNIST as our testbed.

The goal of this walkthrough is not to overwhelm you with implementation details, but to give you a structured path through the code. Follow the recommended file order below to see how the components compose, from pure PyTorch ops, to ATen CUDA calls, to quantizer/swizzler and finally to cuBLASLt MX/NV matmuls. The code is commented in an incremental manner; read it in order, and if something feels unclear, trace backward through the earlier steps.

Recommended steps and notes:
* `main_train_tinyvit_mnist.py`: Entry point to train TinyViT on MNIST using various `Linear` implementations. Check `--impl` argument to switch between different backends. 

* `models/{vit.py,transformer_block.py}`: TinyViT model definition with a single Transformer block.

* `custom.py` subclasses `torch.nn.Linear` and use a custom `torch.autograd.Function` to implement the linear operator using native ops for clear illustration of the 3 GEMMs (no quantization yet) and as a template for custom kernels.

* `cuda_aten.py` and `xops.cpp, aten_mm.cpp` introduce a custom C++/CUDA extension that calls `at::addmm/mm`, demonstrating how to wrap custom CUDA code in PyTorch and integrate it as a module.

* `cublaslt.py` and `cublaslt_mm_fp32bf16.cu`: First step toward cuBLASLt integration, implementing BF16/FP32 matmul. Worth reviewing the CUDA code to see how matmul/compute descriptors are set up and cuBLASLt APIs are launched. Reasonably involved, keep the official docs [handy][doc_cublaslt].

* `quantize.py`: Before using low-precision matmul, we quantize inputs using Microxcaling. Our fork is included as a submodule in this repo. We customize behavior and propagate MX formats downstream for packing/swizzling. Focus `q_mxfp8_rowwise, q_mxfp8_colwise, q_nvfp4_rowwise`. 

* `swizzle.py`: [Layout][swzlayout] transformation of quantization scales for the access patterns required by matmul engine.

* `mxfp8.py` and `cublaslt_mm_mxfp8.cu`: the custom MXFP8 Linear, see how all the pieces come together: quantization (especially blocking axis and configuration), swizzling, cuBLASLt matmul. Good to find out what additional configurations are needed for MXFP8 matmul in cuBLASLt.

* `mxfp8.py` and `cublaslt_mm_nvfp4.cu`: similar to MXFP8 but for NVFP4. Note the different blocking size (16 vs 32) and scale type (FP8 vs E8M0). Also how to pack 2xFP4 values into 1 byte for cuBLASLt.

* `nvf4fwd_mxf8bwd.py`: Implements a hybrid Linear layer with NVFP4 forward and MXFP8 backward using the lower level ops above.

#### TN, NN, NT Layout
*coming soon*

---
### Recent Trends in FP4 Training Research
*coming soon*

---
### Future Plan
- [ ] Supplement quantization CUDA kernel for performance
- [ ] Add training of a TinyGPT on small text dataset

---
### References
1. [Tranformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
1. [Talk in GTC March 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
1. [Pre-training with float8 using torchtitan and torchao](https://docs.pytorch.org/ao/0.12/pretraining.html#pre-training-with-torchtitan)

Blogs:
1. [2025/08/29, Nvidia's on Fine-Tuning gpt-oss with MXFP4/NVFP4 QAT (weight and/or activation)](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/) via [Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss)
1. [2025/08/25, Nvidia's on NVFP4/MXFP4 Training of Mamba-Transformer](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), no code shared yet, most likely via through research kernel like Quartet/Qutlass.
1. [2025/06/05, Nvidia's on FP8 Training via Transformer Engine](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
1. [2025/06/24, Nvidia's on NVFP4 Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
1. [2025/03/13, AMD's blog on FP8 Training](https://rocm.blogs.amd.com/software-tools-optimization/amd-optimized-rocm-docker-for-distributed-training/README.html)

Tailored-made kernels:
1. [2025/08/19, Cursor's on 1.5x Faster MoE Training with Custom MXFP8 Kernels](https://cursor.com/en/blog/kernels#building-the-fastest-mxfp8-quantization-kernel-ever)
2. 2025/08/28, Mojo's Matrix Multiplication on Blackwell, [Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
3. 2025/09/05, Mojo's Matrix Multiplication on Blackwell, [Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul)
4. 2025/09/12, Mojo's Matrix Multiplication on Blackwell, [Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
5. [2023/03/23, Mojo AI's Compute Fragmentation: What Matrix Multiplication Teaches Us](https://www.modular.com/blog/ais-compute-fragmentation-what-matrix-multiplication-teaches-us)

Notes:
1. [Aug State](./notes/250801_State_of_Narrow_Precision.md)
1. See [SOTA Research](./notes/99_sota_notes.md)
1. [Build Transformer Engine from source](https://github.com/vuiseng9/vs-colab/blob/main/cheatsheet/tx-engine.md)

[ocp_mx]: https://www.opencompute.org/documents/
[dsv3]: https://arxiv.org/abs/2412.19437
[blog_nvfp4_i]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[te]:https://github.com/NVIDIA/TransformerEngine
[ghmsmx]: https://github.com/microsoft/microxcaling
[ghmxfork]: https://github.com/vuiseng9/microxcaling/tree/return_quantized
[doc_cublaslt]: https://docs.nvidia.com/cuda/cublas/#narrow-precision-data-types-usage
[swzlayout]: https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
<!-- [vsmx]:  -->