# 1.9 SVD (Idea Only)

**SVD breaks a matrix into pieces ranked by importance, so you can keep the important pieces, drop the rest, and get a much smaller matrix that works almost the same.**

## Why this matters for inference

- It is the math behind "how do we find a good low-rank version of a matrix?" (see 1.8).
- It shows up in model compression and KV cache compression ideas.
- Honest importance: low. Inference engineers rarely compute SVD themselves. Knowing the idea is enough.

## Real-life example

Dal has about 20 ingredients, but most of the taste comes from 3: dal, salt, and tadka. The other 17 (a pinch of hing, one bay leaf, a little jeera) matter only a little.

If you keep only the top 3, the dal tastes about 90% the same and takes far less work.

SVD finds a matrix's "dal, salt, and tadka": the pieces that matter most.

## The math

### Pieces with importance scores

SVD (**S**ingular **V**alue **D**ecomposition) writes any matrix as a sum of simple pieces, each with an importance score:

```
W = piece 1 (score 50) + piece 2 (score 20) + piece 3 (score 0.5) + ...
```

- Big score → important → keep.
- Small score → barely matters → drop.

The scores are called **singular values**. Each piece is a rank-1 matrix (a column × a row, like in 1.8).

### A clean example

```
W = | 1  2  3 |
    | 2  4  6 |   = [1, 2, 3] (as a column) × [1, 2, 3] (as a row)
    | 3  6  9 |
```

This matrix has only one piece. All the other scores are 0, so nothing is lost by keeping just one piece:

```
9 numbers → 6 numbers, exactly the same matrix
```

Real weight matrices aren't this clean. They have many pieces, but often a few big ones and many tiny ones. Dropping the tiny ones gives a close approximation.

### How much space you save

Keep the top `r` pieces of a `(d × d)` matrix:

```
Numbers = 2 × d × r   (instead of d × d)
```

```
d = 4096, keep top 64 pieces:
Full:     4096 × 4096    = 16,777,216
Top 64:   2 × 4096 × 64  =    524,288
→ 32× smaller
```

## Inference connection

| Where | How the idea is used |
|---|---|
| Weight compression | Replace a big matrix with its top pieces to save memory and FLOPs |
| KV cache compression | DeepSeek's MLA stores K and V in a small low-rank form, so the KV cache uses much less memory |
| LoRA | Same low-rank idea, used for fine-tuning changes (1.8) |

- Smaller matrices mean fewer bytes to read and fewer FLOPs, which means faster decode.
- The trade-off: dropping pieces loses some quality. How many pieces to keep is a balance.

## Remember

- SVD splits a matrix into pieces, each with an importance score (singular value).
- Keep the big pieces, drop the small ones → smaller matrix, almost the same result.
- Keeping `r` pieces of a `d × d` matrix needs `2 × d × r` numbers.
- Used in compression, KV cache compression (MLA), and the same idea as LoRA.
- You rarely compute it yourself. The idea is what matters.

## Quick check

1. In the dal example, what plays the role of the singular value?
2. A 1000 × 1000 matrix is reduced to its top 10 pieces. How many numbers remain?
3. Why would a smaller matrix make decode faster?

<details>
<summary>Answers</summary>

1. How much each ingredient affects the taste (dal = big, hing = small).
2. 2 × 1000 × 10 = **20,000** (instead of 1,000,000, 50× smaller).
3. Decode is limited by reading weights from memory. Fewer numbers means fewer bytes to read.

</details>
