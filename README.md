# Inference Engineering Math

The math I need for LLM inference engineering, and nothing more. I'm learning one topic at a time. When I finish a topic, its lesson is added here and linked below.

Each lesson has the same parts: why it matters for inference, a real-life example, a small worked example, the inference connection, key takeaways and a quick check with answers.

✅ = done, 🟡 = in progress, ⬜ = to do, ⭐ = very important

**Progress: 15 done, 22 left, 37 total**

## Section 1: Linear Algebra ✅ (9/9)

- ✅ [Vectors and dot product](01-linear-algebra/01-vectors-and-dot-product.md) ⭐
- ✅ [Matrices and matrix multiplication](01-linear-algebra/02-matrices-and-matrix-multiplication.md) ⭐
- ✅ [Matrix shapes and counting FLOPs](01-linear-algebra/03-matrix-shapes-and-counting-flops.md) ⭐
- ✅ [Linear transformations (+ ReLU filter)](01-linear-algebra/04-linear-transformations.md)
- ✅ [Transpose and identity matrix](01-linear-algebra/05-transpose-and-identity.md)
- ✅ [Tensors: `[batch, tokens, hidden]`](01-linear-algebra/06-tensors.md) ⭐
- ✅ [Splitting matrices across GPUs + Megatron trick](01-linear-algebra/07-splitting-matrices-across-gpus.md) ⭐
- ✅ [Rank and low-rank (LoRA), already known](01-linear-algebra/08-rank-and-low-rank-lora.md)
- ✅ [SVD (idea only)](01-linear-algebra/09-svd.md)

## Section 2: Numbers inside computers (6/7)

- ✅ [Bits and bytes](02-numbers-inside-computers/01-bits-and-bytes.md)
- ✅ [Binary integers (INT8, INT4)](02-numbers-inside-computers/02-binary-integers.md)
- ✅ [Floating point: FP32, FP16, BF16](02-numbers-inside-computers/03-floating-point-fp32-fp16-bf16.md) ⭐
- ✅ [Quantization basics: scale, round, dequantize](02-numbers-inside-computers/04-quantization-basics.md) ⭐
- ✅ [Rounding errors and overflow](02-numbers-inside-computers/05-rounding-errors-and-overflow.md)
- ✅ [FP8](02-numbers-inside-computers/06-fp8.md) ⭐
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
Scale:               scale = biggest number ÷ biggest whole number allowed
Quantize:            stored = round(number ÷ scale)
Dequantize:          back = stored × scale
```

## After the math

1. How a GPU works (HBM, SMs, Tensor Cores, NVLink)
2. Inference techniques (PagedAttention, continuous batching, speculative decoding, FlashAttention, quantization methods like GPTQ, AWQ and SmoothQuant, parallelism)
3. Serving frameworks (vLLM, SGLang, TensorRT-LLM)
4. Measuring and profiling (TTFT, TPOT, throughput, Nsight)
5. Projects
