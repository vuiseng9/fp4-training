


What is the best results in FP4 training, i.e. between MXFP4 (K=32, S=E8) vs NVFP4 (K=16, S=E4M3)

```
25/05/25, FP4 All the Way: Fully Quantized Training of LLMs (Intel, Gaudi2)
25/05/20, Quartet: Native FP4 Training Can Be Optimal for Large Language Models (ISTA, ETH, Red Hat AI)
25/03/04, Training LLMs with MXFP4 (Cornell & AWS AI)
25/01/28, Optimizing Large Language Model Training Using FP4 Quantization (Microsoft Research Asia)
```

* MSRA-FP4 only accelerates one of 3 GEMMs in training because only FP8 forward pass in the training. They replace STE with DGE and * Outlier Clamp Compensation (OCC) which entailing sparse residual matrix implementation.
AWS-MXFP4 also accelerates FP4 forward pass only in training, therefore only 1/3 GEMMs.
* Gaudi-NVFP4 accelerates 3 GEMMs in training, that is forward, backward and updates


Exercise Gaudi-NVFP4 to transformer Engine?

low-precision transformer
check Nvidia Transformer Engine FP8 Recipe uses Optimizer like FP16 training, i.e. momentum, variance, weight in fp32 copy
optimizer states in lower precision is more novel, deepseek does already? where is the codebase?
fit more batch size in single gpu
FSDP even crazier


questions:
1. do we still do scaling at the end to the loss?

| **Datatype**            | **MXFP4** | **NVFP4** |
| ----------------------- | --------- | --------- |
| **Data Representation** | E2M1      | E2M1      |
| **Block Size**          | 32        | 16        |
| **Scale Format**        | E8M0      | E4M3      |
