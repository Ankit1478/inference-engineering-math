# 2.4 Quantization Basics: Scale, Round, Dequantize

**Quantization stores decimal weights as small integers by dividing them by a scale and rounding, then multiplies them back by the scale when they are used.**

## Why this matters for inference

- Quantization is one of the most powerful tools an inference engineer has.
- It makes the model **smaller**, so it fits on fewer GPUs.
- It makes decode **faster**, because there are fewer bytes to read per token.
- It makes serving **cheaper**.
- The cost is a small loss in accuracy, which you must measure.

## Real-life example

Shop prices: ₹49.80, ₹102.30, ₹7.60

| How much you round | Prices | Detail |
|---|---|---|
| To the nearest ₹1 | 50, 102, 8 | Quite good |
| To the nearest ₹10 | 50, 100, 10 | Rough |

Rounding saves effort but loses a bit of detail. The coarser you round, the more you lose.

Quantization is the same: round decimal weights to a small set of integer values.

## The math

### The two formulas

```
Quantize (store):     q = round(w ÷ scale)
Dequantize (use):     w ≈ q × scale
```

### How to choose the scale

```
scale = (largest absolute weight) ÷ (largest integer allowed)
```

- INT8 (symmetric): largest integer = 127
- INT4 (symmetric): largest integer = 7

### Worked example: INT8

Weights (BF16):

```
[0.50, -1.27, 0.034, 0.886]
```

**Step A: find the scale**

```
Largest absolute weight = 1.27
scale = 1.27 ÷ 127 = 0.01
```

**Step B: divide and round**

```
 0.50  ÷ 0.01 =   50    →   50
-1.27  ÷ 0.01 = -127    → -127
 0.034 ÷ 0.01 =    3.4  →    3
 0.886 ÷ 0.01 =   88.6  →   89
```

Store `[50, -127, 3, 89]` (1 byte each) plus the scale `0.01` once.

**Step C: dequantize when needed**

```
  50 × 0.01 =  0.50   (true 0.50)   error 0
-127 × 0.01 = -1.27   (true -1.27)  error 0
   3 × 0.01 =  0.03   (true 0.034)  error 0.004
  89 × 0.01 =  0.89   (true 0.886)  error 0.004
```

Very small errors.

### Worked example: INT4

Same weights, but now only -7 to 7 is allowed.

```
scale = 1.27 ÷ 7 ≈ 0.181

 0.50  ÷ 0.181 ≈  2.76 →  3 →  3 × 0.181 ≈  0.544   (true 0.50)   error 0.044
-1.27  ÷ 0.181 ≈ -7.00 → -7 → -7 × 0.181 ≈ -1.27    (true -1.27)  error 0
 0.034 ÷ 0.181 ≈  0.19 →  0 →  0 × 0.181 =  0.00    (true 0.034)  error 0.034
 0.886 ÷ 0.181 ≈  4.88 →  5 →  5 × 0.181 ≈  0.907   (true 0.886)  error 0.021
```

Bigger errors, but half the size of INT8. Note that the small weight 0.034 became **0**.

### The outlier problem

What if one weight is much bigger than the rest?

```
Weights: [0.02, -0.05, 0.03, 50.0]
scale = 50.0 ÷ 127 ≈ 0.394

0.02  ÷ 0.394 ≈ 0.05 → 0
-0.05 ÷ 0.394 ≈ -0.13 → 0
0.03  ÷ 0.394 ≈ 0.08 → 0
50.0  ÷ 0.394 = 127  → 127
```

One big weight ("outlier") made the scale huge, and all the small weights became **0**. Their information is lost.

### The fix: group-wise scales

Split the weights into **small groups** (for example 128 weights each) and give **each group its own scale**.

- An outlier only spoils the scale of its own group.
- Other groups keep a small scale and good detail.
- Storing extra scales costs a little memory, but it is worth it.

This is how most INT4 methods keep good quality.

## Inference connection

### 70B model sizes and read time

> Note: simplified round numbers are used here: memory bandwidth ≈ 2,000 GB/s, H100 memory = 80 GB. A real H100 SXM is about 3.35 TB/s.

| Format | Size | Fits on one 80 GB GPU? | Read time per token |
|---|---|---|---|
| BF16 | 140 GB | ❌ No (needs 2+) | 70 ms |
| INT8 | 70 GB | ✅ Yes (tight) | 35 ms |
| INT4 | 35 GB | ✅ Yes, with room for the KV cache | 17.5 ms |

### Quality trade-off

| Format | Bytes per weight | Error | Typical quality |
|---|---|---|---|
| BF16 | 2 | none | Original |
| INT8 | 1 | very small | Almost the same |
| INT4 | 0.5 | larger | A little lower, often fine |

Always **measure** quality after quantizing (for example with perplexity, covered in Section 4).

## Remember

- Quantize: `q = round(w ÷ scale)`. Dequantize: `w ≈ q × scale`.
- `scale = max |w| ÷ max integer` (127 for INT8, 7 for INT4).
- INT8 errors are tiny. INT4 errors are larger, but the model is half the size.
- Outliers ruin the scale. Group-wise scales fix this.
- Quantization = smaller + faster + cheaper, with a small quality cost.

## Quick check

1. Weights `[0.2, -0.4, 0.1]`. What is the INT8 scale?
2. Using that scale, what are the stored integers?
3. Why does one large weight hurt quantization?
4. A 13B model in INT4. How big is it?

<details>
<summary>Answers</summary>

1. scale = 0.4 ÷ 127 ≈ **0.00315**
2. 0.2 ÷ 0.00315 ≈ 63.5 → **64** (63 is also acceptable), -0.4 → **-127**, 0.1 ÷ 0.00315 ≈ 31.75 → **32**
3. It makes the scale large, so the small weights round to 0 and lose their information.
4. 13 × 0.5 = **6.5 GB**

</details>
