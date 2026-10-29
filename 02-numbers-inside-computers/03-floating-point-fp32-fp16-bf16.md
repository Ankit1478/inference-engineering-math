# 2.3 Floating Point: FP32, FP16, BF16

**A floating point number is stored like scientific notation, with exponent bits for range and mantissa bits for precision; FP32, FP16 and BF16 only differ in how they split their bits.**

## Why this matters for inference

- Model weights are released in floating point, usually **BF16** (2 bytes).
- Choosing FP16 vs BF16 can be the difference between a working model and one that outputs `NaN` or `inf`.
- Knowing "range vs precision" explains why BF16 became the standard.
- This is one of the most important topics in Section 2.

## Real-life example

In school you wrote big and small numbers like this:

```
1,23,000 = 1.23 × 10^5
0.00075  = 7.5  × 10^-4
```

Each number has **two parts**:

- **Digits** (1.23): how **exact** the number is
- **Power** (10^5): how **big or small** the number is

The decimal point "floats" depending on the power. That is why it is called **floating point (FP)**.

Another way to think about it is a map:

- The **exponent** is the **zoom level** (world, country, city).
- The **mantissa** is the **detail** you can see at that zoom.

## The math

### Computers use powers of 2

Same idea, but with 2 instead of 10:

```
0.75 = 1.5 × 2^-1
6.0  = 1.5 × 2^2
```

### The three parts

```
| Sign | Exponent | Mantissa |
|  +/- |  power   |  digits  |
```

| Part | What it controls |
|---|---|
| Sign (1 bit) | Positive or negative |
| Exponent | **Range**: how big or how small a number can be |
| Mantissa | **Precision**: how exact the number is |

### How the formats split their bits

| Format | Total bits | Sign | Exponent (range) | Mantissa (precision) | Bytes |
|---|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 | 4 |
| FP16 | 16 | 1 | **5** | 10 | 2 |
| BF16 | 16 | 1 | **8** | 7 | 2 |

What that means in practice:

| Format | Largest number | Precision (about) |
|---|---|---|
| FP32 | about 3.4 × 10^38 | about 7 digits |
| FP16 | **65,504** | about 3 digits |
| BF16 | about 3.4 × 10^38 (same as FP32) | about 2 digits |

### Worked example 1: storing 70,000 (range)

```
FP32 → 70,000      ✅ exact
FP16 → inf          ❌ overflow (bigger than 65,504)
BF16 → 70,144       ✅ a little off, but usable
```

FP16 has only 5 exponent bits, so it cannot "zoom out" far enough.

### Worked example 2: storing 1/3 (precision)

```
True value: 0.333333...
FP32 → 0.33333334
FP16 → 0.33325
BF16 → 0.33398
```

BF16 is the least exact, because it has only 7 mantissa bits.

### Worked example 3: storing 0.1

```
FP16 → 0.099976
BF16 → 0.100098
```

Neither is exactly 0.1. Small errors like this are normal (see topic 2.5).

## Inference connection

### Why BF16 is the standard

- Inside an LLM, some numbers become large (for example in attention scores and some activations).
- In **FP16**, a number above 65,504 becomes `inf`, and the model starts producing garbage or `NaN`.
- **BF16** keeps the full FP32 range, so it does not overflow.
- Neural networks are fine with less precision. **Range matters more than precision.**

BF16 means "Brain Float 16". Google Brain designed it by keeping FP32's 8 exponent bits and cutting the mantissa down.

### Size and speed

| Format | 7B model | 70B model |
|---|---|---|
| FP32 | 28 GB | 280 GB |
| FP16 / BF16 | 14 GB | 140 GB |

Going from FP32 to BF16 halves the memory and halves the bytes read per token.

### Debugging tip

If a model gives `NaN` or `inf` outputs when run in FP16, the first suspect is **overflow**. Try BF16.

## Remember

- Floating point = sign + exponent + mantissa.
- Exponent bits = **range**. Mantissa bits = **precision**.
- FP32 = 4 bytes, exact. FP16 = 2 bytes, small range (max 65,504). BF16 = 2 bytes, big range, less precision.
- BF16 is the standard for LLM inference because it does not overflow.
- `NaN`/`inf` in FP16 usually means overflow.

## Quick check

1. Which part of a float decides how big a number can be?
2. What happens if you store 100,000 in FP16?
3. Why do LLMs prefer BF16 over FP16 if both are 2 bytes?
4. How big is a 13B model in FP32 and in BF16?

<details>
<summary>Answers</summary>

1. The **exponent**.
2. It **overflows to inf**, because FP16's maximum is 65,504. (In BF16 it becomes 99,840.)
3. BF16 has the same **range** as FP32, so large values don't overflow. LLMs need range more than precision.
4. FP32: 13 × 4 = **52 GB**. BF16: 13 × 2 = **26 GB**.

</details>
