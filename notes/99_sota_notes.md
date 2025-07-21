


## FP4 Training

| Aspect              | MXFP4      | NVFP4      |
| ---------------     | -----------| -----------|
| Value type **FP4**  | E2M1 | E2M1 |
| Block size          | 32   | 16   |
| Scale type **FP8**  | E8M0 | E4M3 |

<details>
<summary><b>25/05/25, FP4 All the Way: Fully Quantized Training of LLMs (Intel, Gaudi2)</b></summary>
This work is among the first to study fully quantized training in FP4 precision. By fully quantized training (FQT), it means mapping 3 GEMMs involved in forward and backward passes of Linear to FP4 MatMul which are readily available in upcoming MX-compliant HW. The authors investigate the following:
    
* Block Size and Scaler Datatype: Study on training convergence concluded that NVFP4 choice of block size 16 and scaler E4M3 datatype are optimal as compared to other formats E1M6, E2M5, E3M4, E4M3, E5M2, E6M1, E8M0, including MXFP4 (k=32, s=E8M0).
* Rounding scheme of respective quantizer: The 3 GEMMs entail a set of 6 quantization operations. The work questions the use of Round-to-Nearest (RtN) or Stochastic Rounding (SR) in these quantizations. Based on empirical results which aligned to previous studies, authors proposed rounding to respective quantizer, i.e.:
    * Forward Linear GEMM: RtN quantizer both input
    * Backward Grad X GEMM: RtN quantizer for transposed W and SR Quantizer for incoming gradient
    * Backward Grad W GEMM: SR quantizer for both input
* Theoretical analysis on noise threshold to training convergence: the average per-coordinate gradient magnitude falls approximately below √3 times the quantization noise standard deviation, training no longer yields effective loss reduction. Experiments validated the analysis. As a work around, quantize-aware fine tuning is proposed, i.e. backward in higher precision for the last-mile convergence.
</details>

<details>
<summary><b>25/05/20, Quartet: Native FP4 Training Can Be Optimal for Large Language Models (ISTA, ETH, Red Hat AI)</b></summary>
</details>

<details>
<summary><b>25/03/04, Training LLMs with MXFP4 (Cornell & AWS AI)</b></summary>
AWS-MXFP4 also accelerates FP4 forward pass only in training, therefore only 1/3 GEMMs. *More to add...*
</details>

<details>
<summary><b>25/02/28, Oscillation-Reduced MXFP4 Training for Vision Transformers</b></summary>
</details>

<details>
<summary><b>25/01/28, Optimizing Large Language Model Training Using FP4 Quantization (Microsoft Research Asia)</b></summary>
MSRA-FP4 only accelerates one of 3 GEMMs in training because only FP8 forward pass in the training. They replace STE with DGE and * Outlier Clamp Compensation (OCC) which entailing sparse residual matrix implementation. *More to add...*
</details>



## FP8 Training
<details>
<summary><b>25/05/30, Recipes for Pre-training LLMs with MXFP8 (Nvidia)</b></summary>
</details>





------



Potential value added work
* Emulated studies
    * MXFP4 following Mishra's MXFP8 recipe
    * NVFP4 following Gaudi's paper

    * plausible paths
        1) tetrajet, reproducible will be ideal then we start from there
        2) directly from timm
        * Kernel Acceleration
        * map to MI300X
        * blackwell

* To transformer Engine? entailing both recipe and kernel implementation

### Questions
Gaudi-NVFP4
* do they take care of inner dim? other wise they only need to quantize 1

* Optimizer
check Nvidia Transformer Engine FP8 Recipe uses Optimizer like FP16 training, i.e. momentum, variance, weight in fp32 copy
optimizer states in lower precision is more novel, deepseek does already? where is the codebase?
fit more batch size in single gpu
FSDP even crazier


Questions:
1. do we still do gradient scaling at the end to the loss? maybe no, just complicate the scenario?
age old 




mxfp4, nvfp4 simulated quantized training for vision transformers



dev we can use our desktop

how to use huggingface dataset to download