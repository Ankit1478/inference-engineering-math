# Inference Engineering Math

The math I need for LLM inference engineering, and nothing more. I'm learning one topic at a time. When I finish a topic, its lesson is added here and linked below.

Each lesson has the same parts: why it matters for inference, a real-life example, a small worked example, the inference connection, key takeaways and a quick check with answers.

✅ = done, 🟡 = in progress, ⬜ = to do, ⭐ = very important

**Progress: 30 done, 7 left, 37 total**

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

## Section 2: Numbers inside computers ✅ (7/7)

- ✅ [Bits and bytes](02-numbers-inside-computers/01-bits-and-bytes.md)
- ✅ [Binary integers (INT8, INT4)](02-numbers-inside-computers/02-binary-integers.md)
- ✅ [Floating point: FP32, FP16, BF16](02-numbers-inside-computers/03-floating-point-fp32-fp16-bf16.md) ⭐
- ✅ [Quantization basics: scale, round, dequantize](02-numbers-inside-computers/04-quantization-basics.md) ⭐
- ✅ [Rounding errors and overflow](02-numbers-inside-computers/05-rounding-errors-and-overflow.md)
- ✅ [FP8](02-numbers-inside-computers/06-fp8.md) ⭐
- ✅ [What gets quantized: weights, activations, KV cache](02-numbers-inside-computers/07-what-gets-quantized.md) ⭐

## KV Cache deep dive ⭐ ✅ (20/20)

I learned the KV cache properly before going on. I already know transformers, so attention isn't repeated here.

⚡ **[Quick revision: the whole KV cache on one page](kv-cache-deep-dive/00-quick-revision.md)**

**[1. Basics](kv-cache-deep-dive/01-basics.md)** ✅

- ✅ How an LLM writes: one word at a time, looking back at all old words
- ✅ Attention basics: Q, K, V
- ✅ Without cache vs with cache: make K and V once and save them (about 500× less work for 1,000 words)

**[2. Core](kv-cache-deep-dive/02-core.md)** ⭐ ✅

- ✅ Why we save K and V, but not Q
- ✅ Prefill vs decode
- ✅ KV cache size formula: `2 × layers × KV heads × head size × bytes × words × users`
- ✅ Why decode is slow: limited by memory speed, not maths

**[3. Making it smaller](kv-cache-deep-dive/03-making-it-smaller.md)** ⭐ ✅

- ✅ MHA → MQA → GQA
- ✅ FP8 KV cache
- ✅ Sliding window

**[4. Managing it on a real server](kv-cache-deep-dive/04-managing-it-on-a-server.md)** ⭐ ✅

- ✅ The waste problem (fragmentation)
- ✅ PagedAttention (vLLM): small pages, given only when needed, stored anywhere
- ✅ Continuous batching: when one user finishes, the next takes their seat right away
- ✅ Prefix caching: reuse K,V of the same start (saves the question, not the answer)

**[5. Advanced](kv-cache-deep-dive/05-advanced.md)** ✅

- ✅ Offloading to CPU memory: move unused K,V out of the GPU, bring it back later
- ✅ Eviction: which old notes to throw away (LRU: used longest ago goes first)
- ✅ Disaggregation: prefill and decode on different GPUs (the KV cache is sent between them)

**[6. Build it in code](kv-cache-deep-dive/06-build-it-in-code.md)** ✅

- ✅ Attention without a cache in PyTorch ([code](kv-cache-deep-dive/code/01_attention_no_cache.py))
- ✅ Attention with a cache, then compare the speed: same answer, 2–24× faster ([code](kv-cache-deep-dive/code/02_attention_with_cache.py))
- ✅ A simple paged KV cache ([code](kv-cache-deep-dive/code/03_paged_kv_cache.py))

## Section 3: Functions used in AI ✅ (4/4)

- ✅ [Exponent (eˣ) and logarithm (log)](03-functions-used-in-ai/01-exponent-and-logarithm.md)
- ✅ [Softmax](03-functions-used-in-ai/02-softmax.md) ⭐
- ✅ [Activation functions: GELU, SiLU, SwiGLU](03-functions-used-in-ai/03-activation-functions-gelu-silu.md)
- ✅ [Sin and cos (RoPE)](03-functions-used-in-ai/04-sin-cos-rope.md)

## Section 4: Probability and Statistics ✅ (5/5)

- ✅ [Probability basics](04-probability-and-statistics/01-probability-mean-variance.md) ⭐
- ✅ [Mean and variance (LayerNorm / RMSNorm)](04-probability-and-statistics/01-probability-mean-variance.md)
- ✅ [Sampling: temperature, top-k, top-p](04-probability-and-statistics/02-sampling.md) ⭐
- ✅ [Entropy, cross-entropy, perplexity](04-probability-and-statistics/03-entropy-perplexity-kl.md)
- ✅ [KL divergence](04-probability-and-statistics/03-entropy-perplexity-kl.md)

## Section 5: Transformer math ⭐ ✅ (5/5)

I already know transformers, so these are revision notes from the inference side: shapes, parameter counts, FLOPs and how each part is served.

- ✅ [Embeddings (word → vector) and lm_head](05-transformer-math/01-embeddings.md)
- ✅ [Attention formula: `softmax(QKᵀ / √d) × V`](05-transformer-math/02-attention-formula.md) ⭐
- ✅ Multi-head attention, already known
- ✅ [Feed-forward layers](03-functions-used-in-ai/03-activation-functions-gelu-silu.md), already known (SwiGLU in 3.3)
- ✅ KV cache and its size formula ⭐ (see the KV Cache deep dive above)

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
GPU memory needed:   weights + KV cache + about 10% extra
KV cache per word:   2 × layers × KV heads × head size × bytes
KV cache total:      KV cache per word × words × users
Users on a GPU:      (GPU memory − weights − extra) ÷ KV per user
W4A16:               4-bit weights, 16-bit activations
Log rule:            log(a × b) = log a + log b
Softmax:             eˣⁱ ÷ sum of all eˣ  (subtract the max first)
Temperature:         softmax(scores ÷ T)
SwiGLU FFN:          Down( SiLU(Gate·x) × Up·x )  → 3 matrices
Rotate [x, y] by θ:  [x·cos θ − y·sin θ, x·sin θ + y·cos θ]
AND / OR:            P(A and B) = P(A) × P(B),  P(A or B) = P(A) + P(B)
Expected value:      sum of (chance × result)
Std:                 √(average of (x − mean)²)
RMSNorm:             x ÷ √(average of x²) × weight
Min-p cut-off:       min_p × top chance
Spec decode accept:  min(1, target chance ÷ draft chance)
Surprise:            −log(chance)
Perplexity:          e^(average surprise on real text)  (lower = better)
KL(P ‖ Q):           sum of P × log(P ÷ Q)  (0 = same)
Tokens:              1 token ≈ 0.75 English words
lm_head FLOPs:       2 × hidden × vocab per token
Attention:           softmax(QKᵀ / √d) × V
Decode attn FLOPs:   4 × context length × hidden, per layer
```

## After the math

1. How a GPU works (HBM, SMs, Tensor Cores, NVLink)
2. Inference techniques (PagedAttention, continuous batching, speculative decoding, FlashAttention, quantization methods like GPTQ, AWQ and SmoothQuant, parallelism)
3. Serving frameworks (vLLM, SGLang, TensorRT-LLM)
4. Measuring and profiling (TTFT, TPOT, throughput, Nsight)
5. Projects
