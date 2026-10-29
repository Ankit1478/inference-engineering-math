# 1.8 Rank and Low-Rank Matrices (LoRA)

**Some big matrices are secretly made of very few patterns, so they can be stored as two thin matrices, which is the idea behind LoRA.**

## Why this matters for inference

- LoRA adapters are tiny add-ons to a base model. Inference servers often load many of them on one GPU to serve many customised models at once.
- The same "two thin matrices" trick is used to compress the KV cache (DeepSeek's MLA) and model weights.
- Honest importance: medium. You need the idea and one formula, not the theory.

## Real-life example

A class of 30 students all get marks that follow one simple pattern: every student's marks are just "their ability × the subject's difficulty". You don't need to store all 30 × 5 marks. You can store 30 abilities and 5 difficulties and rebuild every mark. That table has **one** pattern, so its rank is 1.

## The math

### Rank = how many truly different patterns a matrix contains

```
W = | 1  2  3 |
    | 2  4  6 |
    | 3  6  9 |
```

Every row is a multiple of the first row:

```
Row 2 = 2 × Row 1
Row 3 = 3 × Row 1
```

So W is built from two small vectors:

```
| 1 |
| 2 |  ×  [1, 2, 3]  =  W
| 3 |
(3 × 1)   (1 × 3)
```

```
Storing W directly: 9 numbers
Storing the two pieces: 3 + 3 = 6 numbers
```

W has **rank 1**: one pattern.

### Low-rank form

A big `(d × d)` matrix can often be approximated as:

```
(d × d) ≈ (d × r) × (r × d)      where r is small
```

Numbers stored:

```
Full:      d × d
Low-rank:  d × r + r × d = 2 × d × r
```

Example with `d = 4096`, `r = 16`:

```
Full:      4096 × 4096    = 16,777,216
Low-rank:  2 × 4096 × 16  =    131,072
→ 128× smaller
```

### LoRA in one formula

Fine-tuning normally changes the whole W. LoRA keeps W frozen and learns only a small low-rank change:

```
W_new = W + B × A
B: (d × r),  A: (r × d),  r small (e.g. 8, 16, 64)
```

For one token, you can compute it without ever building the big matrix:

```
y = x × W  +  (x × B) × A
    big        two tiny multiplications
```

## Inference connection

### Adapter size

LoRA rank 16 on two 4096 × 4096 matrices (for example Q and V) in each of 32 layers:

```
Per matrix: 2 × 4096 × 16 = 131,072 numbers
Total:      131,072 × 2 matrices × 32 layers = 8,388,608 numbers (~8.4 million)
In BF16:    8,388,608 × 2 bytes ≈ 16.8 MB
```

Compare with the 7B base model: 7B × 2 bytes = 14 GB. The adapter is about 1/800 of the model's size.

### Multi-LoRA serving

- One GPU holds the 14 GB base model once.
- Hundreds of small adapters (~17 MB each) sit next to it.
- Each request says which adapter to use. Requests for different adapters can still be batched, because the big `x × W` part is shared.
- vLLM and similar servers support this directly (`--enable-lora`).

### Extra cost

- The extra FLOPs are small: `2 × d × r` per matrix instead of `d × d`.
- The adapter can also be **merged** into W ahead of time (`W + B × A`), so it costs nothing at inference. You lose the ability to swap adapters per request, though.

## Remember

- Rank = number of truly different patterns in a matrix.
- A low-rank matrix can be stored as two thin matrices: `(d × r) × (r × d)`.
- Storage drops from `d × d` to `2 × d × r`.
- LoRA: `W_new = W + B × A`, with W frozen and only the small B and A learned.
- Inference: one base model + many tiny adapters on the same GPU.

## Quick check

1. What is the rank of `| 2 4 |`, `| 1 2 |`?
2. How many numbers does a rank-8 version of a 4096 × 4096 matrix need?
3. Why can many LoRA adapters share one GPU?

<details>
<summary>Answers</summary>

1. **1**, because row 1 = 2 × row 2.
2. 2 × 4096 × 8 = **65,536** (vs 16,777,216, about 256× smaller).
3. The big base model is stored once and shared. Each adapter is only a few MB.

</details>
