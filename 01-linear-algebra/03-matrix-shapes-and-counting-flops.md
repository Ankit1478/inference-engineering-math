# 1.3 Matrix Shapes and Counting FLOPs

**FLOPs count how many small math steps (multiplies and adds) a model does, and for one token an LLM needs about 2 × its number of parameters.**

## Why this matters for inference

- You can't make something faster until you know how much work it does. FLOPs measure that work.
- With one formula, you can estimate how long the math for a token takes on a GPU.
- It is one of the three core skills of inference engineering, together with counting bytes and understanding batching.
- You don't need to multiply matrices by hand. You need quick, rough estimates.

## Real-life example

### The calculator

To work out `2×3 + 4×5` on a calculator:

1. `2 × 3` → 1 step (multiply)
2. `4 × 5` → 1 step (multiply)
3. `6 + 20` → 1 step (add)

Total = 3 math steps = **3 FLOPs**.

**1 FLOP = 1 multiply or 1 add.** (FLOP stands for Floating Point Operation.)

### The kitchen

A restaurant kitchen has two limits:

| Limit | Kitchen | GPU |
|---|---|---|
| Ingredient delivery speed | How fast ingredients arrive | Memory bandwidth (reading W) |
| Cooking speed | How fast the chef cooks | FLOPs per second (doing math) |

With few customers, the chef waits for deliveries (memory is the problem). With a huge crowd, the chef can't cook fast enough (FLOPs are the problem).

## The math

### Step 1: The shape rule

```
(m × k) · (k × n) = (m × n)
      └─must match─┘
```

- The **inner** numbers must be equal, or you can't multiply.
- The **outer** numbers give the output shape.

### Step 2: Count one dot product

A dot product of `k` numbers = `k` multiplies and about `k` adds.

```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6
→ 3 multiplies, ~3 adds, ~6 FLOPs
```

### Step 3: Count a full matrix multiplication

The output has `m × n` cells. Each cell is one dot product of length `k`.

```
Multiplies = m × n × k
FLOPs      = 2 × m × n × k
```

### Worked example

```
| 1  2 |     | 1  2 |     | 7  10 |
| 3  1 |  ×  | 3  4 |  =  | 6  10 |
(2 × 2)     (2 × 2)

m = 2, k = 2, n = 2
Multiplies = 2 × 2 × 2 = 8
FLOPs      = 2 × 8     = 16
```

Check by hand:

```
7  = 1×1 + 2×3  → 2 multiplies
10 = 1×2 + 2×4  → 2 multiplies
6  = 3×1 + 1×3  → 2 multiplies
10 = 3×2 + 1×4  → 2 multiplies
                  total 8
```

### Step 4: One token through one weight matrix

```
Token vector: (1 × 4096)
W:            (4096 × 4096)

Multiplies = 1 × 4096 × 4096 = 16,777,216  (~16.7 million)
FLOPs      = 2 × 16.7 million = ~33.5 million
```

Notice: for one token, the number of multiplies equals the number of weights in W. **Every weight is multiplied exactly once per token.**

### Step 5: The shortcut

Since every weight is used once per token:

```
FLOPs per token ≈ 2 × parameters
FLOPs for B tokens ≈ 2 × parameters × B
```

| Model | Parameters | FLOPs per token |
|---|---|---|
| 3B | 3 billion | ~6 billion |
| 7B | 7 billion | ~14 billion |
| 70B | 70 billion | ~140 billion |

Attention adds a little extra math that grows with conversation length (covered in Section 5). For now the shortcut is enough.

## Inference connection

(Simplified round numbers below: ~1,000 TFLOPS of compute and ~2,000 GB/s of memory bandwidth. A real H100 SXM is ~990 dense BF16 TFLOPS and ~3.35 TB/s.)

### Math time vs read time for one token (7B model)

```
Math time = 14 billion FLOPs ÷ 1,000 trillion FLOPs/s ≈ 0.014 ms
Read time = 14 GB ÷ 2,000 GB/s                        ≈ 7 ms
```

| Part | Time |
|---|---|
| Math (FLOPs) | ~0.014 ms |
| Reading weights | ~7 ms |

The math is about 500× faster than the reading. During decode, the GPU mostly **waits for memory**. That is why batching helps: one read serves many users' math.

### Where FLOPs become the limit

**1. Time before the first word (TTFT, Time To First Token).** A long prompt means a lot of math before the answer starts.

```
7B model, 10,000-token prompt:
FLOPs = 2 × 7B × 10,000 = 140 trillion
At a realistic ~50% of 1,000 TFLOPS: 140 ÷ 500 ≈ 0.28 s
Same prompt on a 70B model: ≈ 2.8 s
```

**2. How many users one GPU can serve.** Read time stays fixed. Math grows with each user.

| Users in batch | Math time | Read time | Bottleneck |
|---|---|---|---|
| 1 | 0.014 ms | 7 ms | Memory |
| 100 | 1.4 ms | 7 ms | Memory |
| 500 | 7 ms | 7 ms | Balanced |
| 1,000 | 14 ms | 7 ms | FLOPs |

These are ideal numbers. In practice the limit comes earlier, but the idea is the same.

**3. Money.** A 70B model needs ~10× the FLOPs of a 7B model, so roughly 10× the GPUs and cost.

## Remember

- 1 FLOP = 1 multiply or 1 add.
- Shape rule: `(m × k) · (k × n) = (m × n)`, inner numbers must match.
- FLOPs for a matrix multiplication = `2 × m × n × k`.
- FLOPs per token ≈ `2 × parameters` (7B → ~14 billion).
- Decode is limited by memory, not math. Long prompts and huge batches are limited by FLOPs.

## Quick check

1. `A (3 × 5) · B (5 × 2)`: how many multiplies and FLOPs?
2. One token `(1 × 4096)` through `W (4096 × 11008)`: how many FLOPs?
3. A 13B model: FLOPs per token?
4. A 7B model processes a 100-token prompt. Total FLOPs?

<details>
<summary>Answers</summary>

1. 3 × 2 × 5 = **30** multiplies, **60** FLOPs. Output shape is 3 × 2.
2. 2 × 1 × 4096 × 11008 = **90,177,536** (~90 million) FLOPs
3. 2 × 13B = **~26 billion** FLOPs
4. 2 × 7B × 100 = **1.4 trillion** FLOPs

</details>
