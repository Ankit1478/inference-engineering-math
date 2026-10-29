# 1.2 Matrices and Matrix Multiplication

**A matrix is a grid of numbers, and multiplying by a matrix means doing many dot products at once; this is almost all of the work an LLM does.**

## Why this matters for inference

- A model's weights (called **W**) are stored as matrices. A "7B model" is about 7 billion numbers spread across many matrices.
- Nearly all of an LLM's GPU time is spent on matrix multiplication.
- Understanding matrix × vector vs matrix × matrix explains why **batching** makes inference much cheaper. This is one of the most important ideas in the whole subject.

## Real-life example

### Many judges at once

In the dot product lesson, one judge scored a phone. A matrix is a **panel of judges**. Each row is one judge with their own priorities:

```
W = | 3  2  0 |   ← Judge 1: "camera lover"
    | 0  1  3 |   ← Judge 2: "style lover"
```

Multiplying the phone `[8, 5, 9]` by this matrix asks every judge for a score in one go.

### The bus

A bus trip from city A to city B costs ₹1000 in fuel. It costs the same whether 1 person or 40 people ride. So it is smart to fill the bus.

For a GPU, **reading the weights from memory is the bus trip**. Each user is a passenger. You will see below why matrix × matrix lets many users share one trip.

## The math

### Shape

A matrix has a shape: **rows × columns**.

```
W = | 1  0  2 |      2 rows, 3 columns → shape 2 × 3
    | 3  1  0 |
```

### Matrix × vector

Do a dot product of each row with the vector. Each row gives one output number.

```
W = | 1  0  2 |     x = [2, 1, 3]
    | 3  1  0 |

Row 1 · x = 1×2 + 0×1 + 2×3 = 8
Row 2 · x = 3×2 + 1×1 + 0×3 = 7

W × x = [8, 7]
```

The phone example:

```
Judge 1 = 8×3 + 5×2 + 9×0 = 34
Judge 2 = 8×0 + 5×1 + 9×3 = 32
```

### Matrix × matrix

Each output cell is a dot product of **a row from the first matrix** with **a column from the second matrix**.

```
A = | 1  2 |     B = | 5  6 |
    | 3  4 |         | 7  8 |

cell [1][1] = row 1 · col 1 = 1×5 + 2×7 = 19
cell [1][2] = row 1 · col 2 = 1×6 + 2×8 = 22
cell [2][1] = row 2 · col 1 = 3×5 + 4×7 = 43
cell [2][2] = row 2 · col 2 = 3×6 + 4×8 = 50

A × B = | 19  22 |
        | 43  50 |
```

Order matters: `A × B` is usually **not** the same as `B × A`.

### Batching, shown with numbers

One weight matrix with 4 numbers:

```
W = | 1  2 |
    | 3  4 |
```

Two users send input vectors:

```
User A = [1, 2]
User B = [3, 1]
```

**Way 1: one user at a time**

```
User A: [1, 2] × W = [1×1 + 2×3, 1×2 + 2×4] = [7, 10]   (read W: 4 numbers)
User B: [3, 1] × W = [3×1 + 1×3, 3×2 + 1×4] = [6, 10]   (read W again: 4 numbers)

Total numbers read from W = 8   (two bus trips)
```

**Way 2: both users together**

Stack the users as rows of one matrix:

```
| 1  2 |   ← User A        | 1  2 |     | 7  10 |   ← User A's answer
| 3  1 |   ← User B   ×    | 3  4 |  =  | 6  10 |   ← User B's answer

Total numbers read from W = 4   (one bus trip)
```

| | One at a time | Together |
|---|---|---|
| Answers | [7,10] and [6,10] | [7,10] and [6,10] (same) |
| Numbers read from W | 8 | 4 |

Same answers, half the memory reads. Users are **not** mixed together: each row stays separate, and each user gets their own answer. Only W is shared.

## Inference connection

An LLM works in two phases:

| Phase | What happens | Operation | Efficiency |
|---|---|---|---|
| **Prefill** (reading your prompt) | All prompt tokens are known, so they go in together | matrix × matrix | Efficient |
| **Decode** (writing the answer) | One token at a time, because each new word depends on the last | matrix × vector | Wasteful |

During decode, every new token needs the GPU to read **all** the weights again.

Rough numbers for a 7B model in BF16 (2 bytes per number):

```
Weights = 7 billion × 2 bytes = 14 GB
Read time = 14 GB ÷ 2,000 GB/s ≈ 7 ms per step
```

(Simplified round numbers: a real H100 SXM reads ~3.35 TB/s and does ~990 dense BF16 TFLOPS. We use 2,000 GB/s and 1,000 TFLOPS to keep the math easy.)

| Users in one batch | Time per step | Tokens per step | Tokens per second |
|---|---|---|---|
| 1 | ~7 ms | 1 | ~140 |
| 32 | ~7 ms (about the same) | 32 | ~4,500 |

Reading the weights costs about the same for 1 user or 32 users, so batching gives roughly 32× more output from the same GPU. This is why inference servers like vLLM group many users' requests into one matrix multiplication.

## Remember

- A matrix is a grid of numbers with shape rows × columns. Model weights (W) are matrices.
- Matrix × vector = one dot product per row.
- Matrix × matrix: each output cell = row of the first · column of the second.
- Decode (one token) = matrix × vector. Prefill and batches = matrix × matrix.
- Batching lets many users share one read of the weights, like passengers sharing one bus trip.

## Quick check

1. W has rows `[2, 1]` and `[0, 3]`, and `x = [1, 4]`. Compute `W × x`.
2. W has 4 numbers and 5 users are batched together. How many numbers are read from W?
3. A model writes a 100-word answer with 10 users batched. How many times does the GPU read all the weights?
4. Why is decode less efficient than prefill?

<details>
<summary>Answers</summary>

1. Row 1: 2×1 + 1×4 = 6. Row 2: 0×1 + 3×4 = 12. Answer: **[6, 12]**
2. **4**. W is read once and shared by all 5 users.
3. **100**. One full read per generated word, shared by all 10 users.
4. Decode does one token at a time (matrix × vector), so the GPU reads all the weights to do only a little math.

</details>
