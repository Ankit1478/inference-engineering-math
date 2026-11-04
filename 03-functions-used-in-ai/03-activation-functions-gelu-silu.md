# 3.3 Activation functions: GELU, SiLU

**An activation function is a small filter applied to every number after a matrix multiply. GELU and SiLU are smooth versions of ReLU.**

## Quick summary (read this first)

1. **Activation** = a filter on each number, used in the **FFN** between two matrix multiplies.
2. **ReLU** (done in 1.4): negative → 0, positive stays the same.
3. **GELU** and **SiLU** = **smooth ReLU**. Small negatives become a **little** negative instead of exactly 0.
4. **SiLU** = `x × sigmoid(x)`. Llama, Mistral and Qwen use it inside **SwiGLU**.
5. **SwiGLU** FFN has **3 weight matrices** instead of 2.
6. **For inference:** the activation is cheap maths, so engines **fuse** it into the matrix multiply to avoid an extra memory trip.

## Why this matters for inference

- You only need the **idea**, not the formula details.
- SwiGLU's 3 matrices mean the **FFN holds about 70% of the weights**, so it's the biggest part to quantize and read.
- Activations are a classic case for **kernel fusion**: don't write the numbers to memory and read them back.

## Real-life example

### A water filter 🚰

| Filter | What it does |
|---|---|
| **ReLU** | A hard gate: below 0 is fully blocked, above 0 passes |
| **GELU / SiLU** | A soft gate: a tiny bit of negative leaks through, and the change is smooth |

The soft gate trains better, which is why newer models use it.

## The math

### Side by side

| x | ReLU | GELU | SiLU |
|---|---|---|---|
| -3 | 0 | -0.004 | -0.14 |
| -1 | 0 | -0.16 | -0.27 |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0.84 | 0.73 |
| 3 | 3 | 3.00 | 2.86 |

👉 For big positive numbers, all 3 are about the same. They differ only **near 0**.

### Formulas (just recognise them)

```
sigmoid(x) = 1 ÷ (1 + e⁻ˣ)       squashes any number to between 0 and 1
SiLU(x)    = x × sigmoid(x)
GELU(x)    ≈ x × (chance that a random normal number is below x)
```

### SwiGLU: the FFN in Llama ⭐

**Old FFN** (GPT-2, uses GELU), 2 matrices:

```
out = Down( GELU( Up·x ) )
```

**SwiGLU FFN** (Llama, Mistral, Qwen), 3 matrices:

```
out = Down( SiLU(Gate·x) × (Up·x) )
             ↑ the "gate"   ↑ the content
```

The gate decides **how much** of each content number passes through.

## Inference connection

### The FFN is most of the model

Llama 3 8B: hidden = 4,096, FFN size = 14,336, 32 layers.

```
FFN weights per layer = 3 × 4,096 × 14,336 = 176 million
× 32 layers           = 5.6 billion  →  about 70% of the 8B
```

👉 Most of the weights you read during decode, and most of what you quantize, is the **FFN**.

### Kernel fusion

The activation is **cheap maths** but would need a **full trip to memory**:

```
Without fusion:  matmul → write to memory → read → SiLU → write → read → multiply
With fusion:     matmul → SiLU and multiply right away → write once ✅
```

Memory is the slow part (2.1, KV cache step 7), so engines like vLLM use **fused kernels** (e.g. `SiluAndMul`).

## Remember

- Activation = a filter on each number, between FFN matrices.
- **GELU / SiLU = smooth ReLU**. They only differ near 0.
- Llama uses **SwiGLU** = SiLU gate × content → **3 matrices**.
- The FFN is about **70% of the weights**.
- Activations are cheap maths, so **fuse** them to save memory trips.

## Quick check

1. ReLU(-2) and SiLU(-2): which one is exactly 0?
2. How many weight matrices are in a SwiGLU FFN?
3. Roughly what share of Llama 3 8B's weights is in the FFN?
4. Why fuse the activation into the matmul instead of running it separately?

<details>
<summary>Answers</summary>

1. **ReLU(-2) = 0.** SiLU(-2) ≈ -0.24.
2. **3**: Gate, Up and Down.
3. About **70%**.
4. To avoid an extra **write to and read from memory**. The maths is cheap, but the memory trip is slow.

</details>
