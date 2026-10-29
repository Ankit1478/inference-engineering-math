# 1.6 Tensors

**A tensor is a box of numbers with any number of dimensions, and reading tensor shapes like `[batch, tokens, hidden]` is a daily skill in inference code.**

## Why this matters for inference

- Every input, output, weight, and cache inside a model is a tensor. Inference code is full of shapes.
- From a shape alone you can tell how much memory a tensor uses and whether the model is in prefill or decode.
- When code breaks with a "shape mismatch" error, this is the knowledge you need to fix it.
- Honest importance: high and very practical.

## Real-life example

A library:

| Thing | Dimensions | Math name |
|---|---|---|
| One line of text | 1D | Vector |
| One page (many lines) | 2D | Matrix |
| One book (many pages) | 3D | Tensor |
| One shelf (many books) | 4D | Tensor |

Vectors and matrices are tensors too. "Tensor" is just the general name.

## The math

### Shape

The shape says how many items there are in each direction.

```
Vector  [3]        → 3 numbers
Matrix  [2, 3]     → 2 rows, 3 columns
Tensor  [2, 3, 4]  → 2 blocks, each with 3 rows of 4 numbers
```

### Total numbers = multiply the shape

```
[2, 3, 4] → 2 × 3 × 4 = 24 numbers
```

### The most common LLM shape

```
[batch, tokens, hidden]
```

| Part | Meaning | Example |
|---|---|---|
| batch | How many users at once | 2 |
| tokens | How many tokens per user | 3 |
| hidden | Size of each token's vector | 4 (4096 in a real 7B model) |

Example with shape `[2, 3, 4]`:

```
User A:
  "What"  → [0.1, 0.5, 0.2, 0.9]
  "is"    → [0.3, 0.1, 0.7, 0.4]
  "AI"    → [0.8, 0.2, 0.1, 0.6]

User B:
  "Hi"    → [0.4, 0.4, 0.3, 0.2]
  "there" → [0.9, 0.1, 0.5, 0.3]
  "bro"   → [0.2, 0.7, 0.6, 0.1]

2 users × 3 tokens × 4 numbers = 24 numbers
```

### Tensor × weight matrix

Only the **last** dimension must match the weight matrix:

```
Input:   [batch, tokens, 4096]
W:                [4096, 11008]
Output:  [batch, tokens, 11008]
```

- `batch` and `tokens` stay the same.
- Only the last number changes.
- The same W is applied to every user and every token. That is batching.

### Memory used by a tensor

```
Memory = total numbers × bytes per number
```

In BF16, 1 number = 2 bytes.

```
Shape [32, 1000, 4096]
Numbers = 32 × 1000 × 4096 = 131,072,000  (~131 million)
Memory  = 131,072,000 × 2  = 262,144,000 bytes ≈ 262 MB
```

## Inference connection

### Prefill vs decode is visible in the shape

| Phase | Shape | Meaning |
|---|---|---|
| Prefill (reading the prompt) | `[batch, 500, 4096]` | 500 tokens at once |
| Decode (writing the answer) | `[batch, 1, 4096]` | One token at a time |

In decode, `tokens = 1`. That is why the GPU does little math and mostly waits for memory.

### Common shapes you will meet

| Tensor | Shape |
|---|---|
| Input token IDs | `[batch, tokens]` |
| Hidden vectors | `[batch, tokens, hidden]` |
| Weight W | `[in, out]` (a 2D matrix) |
| Attention scores | `[batch, heads, tokens, tokens]` |
| KV cache | `[layers, 2, batch, heads, tokens, head_dim]` |

Heads and the KV cache are covered in Section 5. The shapes look big, but you read them the same way.

### In PyTorch

```python
x.shape
# torch.Size([2, 3, 4096])  → 2 users, 3 tokens, 4096 numbers each
```

### "Will it fit?"

Inference engineers constantly ask: will this tensor fit in GPU memory? The answer always comes from `shape → total numbers → × bytes`.

## Remember

- A tensor is a box of numbers: vector (1D), matrix (2D), or more dimensions.
- The most common LLM shape is `[batch, tokens, hidden]`.
- Total numbers = multiply all the numbers in the shape. Memory = total × bytes per number.
- Tensor × W: only the last dimension must match, and only it changes.
- Decode has `tokens = 1`. Prefill has many tokens.

## Quick check

1. How many numbers are in shape `[4, 10, 8]`?
2. `[8, 20, 4096]` × W `[4096, 1024]` gives what shape?
3. Shape `[16, 1, 4096]`: prefill or decode?
4. How much memory does `[1, 2048, 4096]` use in BF16?

<details>
<summary>Answers</summary>

1. 4 × 10 × 8 = **320**
2. **[8, 20, 1024]**
3. **Decode**, because tokens = 1.
4. 1 × 2048 × 4096 = 8,388,608 numbers × 2 bytes = **16,777,216 bytes ≈ 16.8 MB**

</details>
