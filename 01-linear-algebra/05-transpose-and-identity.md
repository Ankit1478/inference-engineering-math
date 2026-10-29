# 1.5 Transpose and Identity Matrix

**Transpose flips a matrix so rows become columns (the numbers stay the same), and the identity matrix is the "do nothing" matrix, like multiplying by 1.**

## Why this matters for inference

- Transpose is what makes the attention formula `Q × Kᵀ` work. Without it, words can't score each other.
- PyTorch stores weights transposed, so shapes in model files often look "backwards". Knowing transpose stops this from confusing you.
- Identity is a small topic. It helps you understand residual connections and is handy for testing code.

## Real-life example

### Transpose: books on a shelf

Books lying flat on a shelf, then stood upright. The content of each book hasn't changed. Only the way they are arranged has.

Or an Excel sheet with students as rows and subjects as columns. Transposing gives subjects as rows and students as columns. Same data, different arrangement.

### Identity: multiplying by 1

`1 × 5 = 5`. Multiplying by 1 changes nothing. The identity matrix does the same for vectors.

## The math

### Transpose (ᵀ)

Rows become columns:

```
A = | 1  2  3 |          Aᵀ = | 1  4 |
    | 4  5  6 |               | 2  5 |
                              | 3  6 |
   (2 × 3)                    (3 × 2)
```

Shape `(m × n)` becomes `(n × m)`.

**Transpose does not change any numbers.** Each row of A is still there, just standing as a column:

```
K = | 1  1  0 |  ← river's vector (lying as a row)
    | 0  0  1 |  ← bank's vector

Kᵀ = | 1  0 |
     | 1  0 |
     | 0  1 |
       ↑  ↑
    river bank   (standing as columns)
```

River is still `[1, 1, 0]`. Bank is still `[0, 0, 1]`. The meaning of each word is untouched.

Transpose is **not** rotation. Rotation (like RoPE) changes the numbers. Transpose only rearranges them.

| | Transpose | Rotation (RoPE) |
|---|---|---|
| What happens | Rows ↔ columns | The vector's numbers change |
| Numbers change? | No | Yes |
| Purpose | Make shapes fit for multiplication | Add word position information |

### Why we need it: matrix multiplication is row × column

Matrix multiplication always pairs **rows of the first** with **columns of the second**. In attention, we want each word's Q (a row) dotted with each word's K (also a row). So we turn K's rows into columns first.

### Worked example: 2 words, vectors of size 3

```
Q = | 1  0  1 |  ← river
    | 1  1  0 |  ← bank
    (2 × 3)

K = | 1  1  0 |  ← river
    | 0  0  1 |  ← bank
    (2 × 3)
```

Try `Q × K`:

```
(2 × 3) · (2 × 3) → inner numbers 3 ≠ 2 → can't multiply
```

Use `Kᵀ` (3 × 2):

```
Q × Kᵀ = (2 × 3) · (3 × 2) = (2 × 2)

river·river = 1×1 + 0×1 + 1×0 = 1
river·bank  = 1×0 + 0×0 + 1×1 = 1
bank·river  = 1×1 + 1×1 + 0×0 = 2
bank·bank   = 1×0 + 1×0 + 0×1 = 0

            river  bank
river  →  |  1     1  |
bank   →  |  2     0  |   ← bank finds river most useful
```

The result is a table of **every word's score with every other word**.

### Identity matrix (I)

1s on the diagonal, 0s everywhere else:

```
I = | 1  0 |
    | 0  1 |

| 1  0 |     | 5 |     | 1×5 + 0×7 |     | 5 |
| 0  1 |  ×  | 7 |  =  | 0×5 + 1×7 |  =  | 7 |
```

`I × x = x`. The same vector comes back.

## Inference connection

### Attention

The attention formula is:

```
softmax(Q × Kᵀ / √d) × V
```

For a prompt with `T` tokens and vector size `d`:

```
Q:  (T × d)
Kᵀ: (d × T)
Q × Kᵀ: (T × T)   ← every token's score with every token
```

For 10 tokens and `d = 64`, the score table is 10 × 10. For 10,000 tokens it is 10,000 × 10,000 = 100 million scores. This is why long prompts make attention expensive (Section 5).

### PyTorch code

PyTorch's `nn.Linear` stores W flipped and computes:

```python
y = x @ W.T    # .T means transpose
```

So a weight file may show `W.shape = (11008, 4096)` even though the layer maps 4096 numbers to 11008. Not a bug: it's the transpose.

### Speed (just be aware)

- GPU memory stores numbers in one long line. Reading them in order is fast; jumping around is slow.
- Sometimes a transpose forces data to be read out of order, which can cost speed.
- Good inference engines (vLLM, TensorRT-LLM) store data in the direction the kernels need.

### Identity

- **Residual connection:** each layer computes `output = x + layer(x)`. The `x +` path lets the original vector pass through unchanged, like an identity path, and the layer only adds new information.
- **Testing:** if multiplying by I doesn't return the same input, your code has a bug.

## Remember

- Transpose = flip rows ↔ columns. `(m × n)` → `(n × m)`. Numbers and meaning stay the same.
- Transpose is rearranging, not rotating.
- `Q × Kᵀ` gives every word's score with every other word. Transpose makes the shapes fit.
- PyTorch computes `y = x @ W.T`, so weight shapes can look reversed.
- Identity: `I × x = x`, the "do nothing" matrix. Related to residual connections.

## Quick check

1. A is (5 × 2). What is the shape of Aᵀ?
2. Q is (10 × 64) and K is (10 × 64). Can you compute Q × K? What shape is Q × Kᵀ?
3. Does transposing K change the meaning of each word's vector?
4. What is `I × [3, 9]`?

<details>
<summary>Answers</summary>

1. **(2 × 5)**
2. Q × K: **no**, because 64 ≠ 10. Q × Kᵀ = **(10 × 10)**, every word's score with every word.
3. **No.** The numbers stay the same; they are just arranged as columns instead of rows.
4. **[3, 9]**

</details>
