# Inference Engineering Math

The math I need for LLM inference engineering, and nothing more. I'm learning one topic at a time and ticking it off here.

✅ = done, ⬜ = to do, ⭐ = very important

**Progress: 13 done, 24 left, 37 total**

## Section 1: Linear Algebra ✅ (9/9)

- ✅ Vectors and dot product ⭐
- ✅ Matrices and matrix multiplication ⭐
- ✅ Matrix shapes and counting FLOPs ⭐
- ✅ Linear transformations (+ ReLU filter)
- ✅ Transpose and identity matrix
- ✅ Tensors: `[batch, tokens, hidden]` ⭐
- ✅ Splitting matrices across GPUs + Megatron trick ⭐
- ✅ Rank and low-rank (LoRA), already known
- ✅ SVD (idea only)

## Section 2: Numbers inside computers (4/7)

- ✅ Bits and bytes
- ✅ Binary integers (INT8, INT4)
- ✅ Floating point: FP32, FP16, BF16 ⭐
- ✅ Quantization basics: scale, round, dequantize ⭐
- ⬜ Rounding errors and overflow
- ⬜ FP8 ⭐
- ⬜ What gets quantized: weights, activations, KV cache ⭐

## Section 3: Functions used in AI (0/4)

- ⬜ Exponent (eˣ) and logarithm (log)
- ⬜ Softmax ⭐
- ⬜ Activation functions: GELU, SiLU (ReLU done)
- ⬜ Sin and cos (RoPE)

## Section 4: Probability and Statistics (0/5)

- ⬜ Probability basics ⭐
- ⬜ Mean and variance (LayerNorm / RMSNorm)
- ⬜ Sampling: temperature, top-k, top-p ⭐
- ⬜ Entropy, cross-entropy, perplexity
- ⬜ KL divergence

## Section 5: Transformer math ⭐ (0/5)

- ⬜ Embeddings (word → vector)
- ⬜ Attention formula: `softmax(QKᵀ / √d) × V` ⭐
- ⬜ Multi-head attention
- ⬜ Feed-forward layers (mostly done)
- ⬜ KV cache and its size formula ⭐

## Section 6: Performance math ⭐ (0/7)

- ⬜ Big-O
- ⬜ Counting FLOPs ⭐ (mostly done)
- ⬜ Counting memory in bytes ⭐ (mostly done)
- ⬜ Memory bandwidth vs compute speed
- ⬜ Arithmetic intensity and the roofline model ⭐
- ⬜ Latency vs throughput ⭐
- ⬜ Basic queuing math

## Section 7: Calculus (skipped)

Only needed for training, not inference.

## Key formulas so far

```
Matrix shape rule:   (m × k) · (k × n) = (m × n)
FLOPs (matmul):      2 × m × n × k
FLOPs per token:     ≈ 2 × parameters
Model size (bytes):  parameters × bytes per number
Read time:           model size ÷ memory bandwidth
Quantize:            stored = round(weight ÷ scale)
Dequantize:          weight ≈ stored × scale
```

## After the math

1. How a GPU works (HBM, SMs, Tensor Cores, NVLink)
2. Inference techniques (PagedAttention, continuous batching, speculative decoding, FlashAttention, quantization methods, parallelism)
3. Serving frameworks (vLLM, SGLang, TensorRT-LLM)
4. Measuring and profiling (TTFT, TPOT, throughput, Nsight)
5. Projects
