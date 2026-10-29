# 2.2 Binary Integers (INT8, INT4)

**A whole number is stored in binary, where each bit says whether one "note" of value 1, 2, 4, 8, ... is used; INT8 and INT4 are the small integer formats used in quantization.**

## Why this matters for inference

- Quantized models store weights as small integers (INT8 or INT4).
- To understand quantization, you need to know what range of values these formats can hold.
- You will rarely convert numbers to binary by hand. Knowing the **ranges** is the important part.

## Real-life example

You have exactly one of each of these notes: **₹8, ₹4, ₹2, ₹1**.

You need to pay **₹13**:

```
₹8 ✅ + ₹4 ✅ + ₹2 ❌ + ₹1 ✅ = 13
  1      1      0      1
```

So **13 = `1101`**.

That is binary. Each bit says "this note is used" (1) or "not used" (0).

## The math

### Place values

Every position is worth double the one on its right:

```
Position value:  128  64  32  16   8   4   2   1
```

### Worked examples (8 bits)

| Number | Binary | How |
|---|---|---|
| 5 | `00000101` | 4 + 1 |
| 6 | `00000110` | 4 + 2 |
| 13 | `00001101` | 8 + 4 + 1 |
| 255 | `11111111` | 128+64+32+16+8+4+2+1 |

### Reading binary back to a number

```
00001101
    8 4 _ 1  → 8 + 4 + 1 = 13
```

### Unsigned (only positive)

With 8 bits you get 256 values:

```
0 to 255
```

### Signed (positive and negative)

To store negative numbers, the values are split between negative and positive. 8 bits still give 256 values, but now:

```
INT8: -128 to 127   (128 negative + 0 + 127 positive = 256 values)
INT4:   -8 to 7     (8 negative + 0 + 7 positive = 16 values)
```

> In quantization, people often use a **symmetric** range and skip the most negative value: INT8 uses -127 to 127, and INT4 uses -7 to 7. That way positive and negative sides are equal.

### The big limitation

Integers have **no decimal point**. You cannot store 0.75 in INT8.

But model weights look like `0.12`, `-0.53`, `0.034`. So how do we store them as integers? That is what **quantization** solves (topic 2.4), using a scale.

## Inference connection

| Format | Bits | Values | Range (signed) | Bytes per weight |
|---|---|---|---|---|
| INT8 | 8 | 256 | -128 to 127 | 1 |
| INT4 | 4 | 16 | -8 to 7 | 0.5 |

- An INT4 weight has only **16 possible values**. Still, quantized INT4 models often work well, thanks to smart scaling.
- Two INT4 weights are packed into one byte. That is why INT4 is "0.5 bytes" per weight.
- GPUs have special hardware for INT8 math, so INT8 can also make the math faster, not only smaller.

Example: a 7B model

```
INT8: 7B × 1   = 7 GB
INT4: 7B × 0.5 = 3.5 GB
```

## Remember

- Binary uses place values 1, 2, 4, 8, 16, ... (each one doubles).
- 13 = 8 + 4 + 1 = `1101`.
- INT8 = -128 to 127 (256 values). INT4 = -8 to 7 (16 values).
- Integers cannot store decimals. Quantization uses a scale to get around this.
- INT4 packs two weights into one byte.

## Quick check

1. Write 6 in 4-bit binary.
2. What number is `1011`?
3. How many different values can INT4 store?
4. Can you store 2.5 directly in INT8?

<details>
<summary>Answers</summary>

1. 4 + 2 = **`0110`**
2. 8 + 2 + 1 = **11**
3. 2^4 = **16** values
4. **No.** Integers have no decimal part. You would need quantization with a scale.

</details>
