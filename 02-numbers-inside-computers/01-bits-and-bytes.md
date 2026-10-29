# 2.1 Bits and Bytes

**A bit is one on/off switch, a byte is 8 of them, and the number of bytes per number decides how big a model is and how fast it runs.**

## Why this matters for inference

- Every model weight is stored in bits. The number of bits per weight decides the model's size in memory.
- Model size decides how many GPUs you need.
- Model size also decides decode speed, because every generated token has to read all the weights from memory.
- This topic is short, but everything else in Section 2 builds on it.

## Real-life example

Think of light bulbs.

**1 bulb** can show 2 different signals:

```
OFF → 0
ON  → 1
= 2 options
```

**2 bulbs** can show 4 different signals:

```
OFF OFF → 00
OFF ON  → 01
ON  OFF → 10
ON  ON  → 11
= 4 options
```

**3 bulbs** can show 8 different signals.

Every new bulb **doubles** the number of options.

A computer works the same way. Inside, it only understands ON and OFF. One switch is called a **bit**.

## The math

### Bit

```
1 bit = one switch = 0 or 1
```

### How many values can n bits store?

```
Options = 2 × 2 × 2 × ... (n times) = 2^n
```

| Bits | Different values | Where you see it |
|---|---|---|
| 1 | 2 | |
| 2 | 4 | |
| 4 | 16 | INT4 |
| 8 | 256 | 1 byte, INT8, FP8 |
| 16 | 65,536 | 2 bytes, FP16, BF16 |
| 32 | about 4.3 billion | 4 bytes, FP32 |

### Byte

```
1 byte = 8 bits = 2^8 = 256 different values
```

A byte is just a name for 8 bits, the same way "a dozen" is a name for 12.

### Worked example

How many values can 3 bits store?

```
2^3 = 2 × 2 × 2 = 8
```

How many bits are in 2 bytes?

```
2 bytes × 8 bits = 16 bits
```

## Inference connection

### More bytes = more detail, but more memory

Think about writing a price:

| How you write ₹123.456789 | Space used | Detail |
|---|---|---|
| ₹123.456789 | More | Very exact |
| ₹123.46 | Less | Good enough |
| ₹123 | Even less | Rough |

Numbers in a computer are the same. More bits let you write a number more exactly. Fewer bits save space.

### Model size formula

```
Model size (bytes) = number of parameters × bytes per parameter
```

For a **70B** model (70 billion parameters):

| Format | Bytes per number | Model size |
|---|---|---|
| FP32 | 4 | 70B × 4 = **280 GB** |
| BF16 | 2 | 70B × 2 = **140 GB** |
| INT8 | 1 | 70B × 1 = **70 GB** |
| INT4 | 0.5 | 70B × 0.5 = **35 GB** |

The parameters are fixed. **We choose the bytes.** The same model can be 280 GB or 35 GB.

### Why size changes speed

> Note: in these notes we use simplified round numbers: GPU memory bandwidth ≈ 2,000 GB/s, compute ≈ 1,000 TFLOPS, H100 memory = 80 GB. A real H100 SXM is about 3.35 TB/s and about 990 dense BF16 TFLOPS.

During decode, the GPU reads every weight once per token.

```
Read time per token = model size ÷ memory bandwidth

70B in BF16: 140 GB ÷ 2,000 GB/s = 0.070 s = 70 ms
70B in INT4:  35 GB ÷ 2,000 GB/s = 0.0175 s = 17.5 ms
```

Fewer bytes means less data to read, so each token is faster.

### Fun scale fact

A 70B model in BF16 is:

```
70 billion × 16 bits = 1.12 trillion switches
```

## Remember

- 1 bit = one switch (0 or 1).
- 1 byte = 8 bits = 256 different values.
- Each extra bit doubles the number of values.
- Model size = parameters × bytes per parameter.
- Fewer bytes = smaller model, faster decode, but less detail per number.

## Quick check

1. How many values can 5 bits store?
2. How many bits are in 4 bytes?
3. How big is a 13B model in BF16? Does it fit on a 24 GB GPU?
4. A 7B model in INT8. How big is it, and how long does one decode step take to read it at 2,000 GB/s?

<details>
<summary>Answers</summary>

1. 2^5 = **32**
2. 4 × 8 = **32 bits**
3. 13B × 2 = **26 GB**. No, it does not fit in 24 GB.
4. 7B × 1 = **7 GB**. 7 ÷ 2,000 = 0.0035 s = **3.5 ms**.

</details>
