# 5.2 Attention formula: softmax(QKᵀ / √d) × V ⭐

**Attention lets every token look back at earlier tokens and take a weighted mix of their information. The formula is 3 steps: score (QKᵀ), turn the scores into weights (softmax), mix (× V).**

## Quick summary (read this first)

1. **Q** = what I'm looking for. **K** = what each token offers. **V** = the information each token carries (KV cache step 2).
2. **Step 1, score:** `Q × Kᵀ` = a dot product (1.1) between my question and every token's label.
3. **Step 2, scale:** divide by **√d** (d = head size, 128 → √128 ≈ 11.3) so the scores don't get too big.
4. **Step 3, mask:** future tokens get **−∞**, so they get weight 0 (causal mask).
5. **Step 4, softmax** (3.2): scores → weights that add up to 1.
6. **Step 5, mix:** `weights × V` = a weighted average of the information.
7. **For inference:** in decode Q is **1 row**, so the cost grows with **context length**. In prefill the score matrix is **n × n**, which is why FlashAttention exists.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| √d is there for accuracy | It's there so softmax doesn't become **all-or-nothing**. |
| Attention cost is fixed per token | In decode it grows with **context length** (reading the whole KV cache). |
| Prefill stores the full n × n score matrix | **FlashAttention** never stores it: it uses chunks plus online softmax (3.2). |
| The mask is applied after softmax | It's applied **before**: −∞ → e^(−∞) = 0. |

## Why this matters for inference

- Attention is the only part of the model whose cost depends on **how long the chat is**.
- Decode attention = reading the KV cache, which is **memory-bound** (KV cache step 7).
- Prefill attention = an **n × n** score matrix, so cost grows with **n²**.
- Long-context serving, FlashAttention, PagedAttention and sliding windows all exist because of this formula.

## Real-life example

### Asking a room for help 🙋

You walk into a room with a **question** (Q). Everyone wears a **name badge** (K) and holds a **notebook** (V).

| Step | Real life | Formula |
|---|---|---|
| 1 | Compare your question with each badge | `Q·K` scores |
| 2 | Calm down over-excited scores | ÷ √d |
| 3 | Ignore people who haven't arrived yet | causal mask |
| 4 | Decide how much to trust each person (shares add up to 100%) | softmax |
| 5 | Mix their notebooks by those shares | × V |

## The math

### 1. Shapes for one head (d = 128, n tokens so far)

```
Q:          [n × 128]
Kᵀ:         [128 × n]
Q × Kᵀ:     [n × n]       ← every token's score against every token
softmax:    [n × n]       (each row adds up to 1)
× V [n×128] → [n × 128]   ← new information for each token
```

**In decode** only the new token asks, so Q is **[1 × 128]**:

```
Q [1×128] × Kᵀ [128×n] → scores [1 × n] → softmax → × V [n×128] → [1 × 128]
```

### 2. Worked example (d = 2, 3 tokens)

```
Q (new token)  = [1, 0]
K: token A = [2, 0], token B = [0, 2], token C = [1, 1]
V: token A = [10, 0], token B = [0, 10], token C = [5, 5]

Scores Q·K:   A = 2, B = 0, C = 1
÷ √2 (1.41):  A = 1.41, B = 0, C = 0.71
softmax:      A = 0.58, B = 0.14, C = 0.28   (adds up to 1)
Mix V:        0.58×[10,0] + 0.14×[0,10] + 0.28×[5,5]
            = [5.8+0+1.4, 0+1.4+1.4] = [7.2, 2.8]
```

The output looks most like A's notebook, because A matched the question best.

### 3. Why divide by √d?

A dot product adds up **d** products, so it gets bigger as d grows (its spread is about √d). Big scores make softmax **all-or-nothing** (3.2):

```
Scores [20, 10]           → softmax [0.99995, 0.00005]   ❌ ignores B completely
÷ √128 → [1.77, 0.88]     → softmax [0.71, 0.29]          ✅ a real mix
```

### 4. Causal mask

A token may only look at itself and **earlier** tokens:

```
Scores for token 2 of 3:  [s1, s2, s3]  → set s3 = −∞ → e^(−∞) = 0
```

### 5. Cost

**Decode, per new token, per layer** (all heads together, hidden = 4,096):

```
Q·Kᵀ: 2 × n × 4,096    +    weights × V: 2 × n × 4,096
= 4 × n × 4,096 FLOPs
```

Llama 3 8B at n = 8,192 tokens:

```
4 × 8,192 × 4,096 = 134M per layer × 32 layers = 4.3 GFLOPs per token
```

Compare that to ~16 GFLOPs for the weights (2 × 8B). At long context, attention is a **big extra**, and it grows with every token.

**Prefill:** the score matrix is n × n per head:

```
n = 8,192 → 67 million scores per head per layer → 134 MB in BF16
```

32 heads × 32 layers of that would be huge. That's why FlashAttention never stores it.

## Inference connection

| Point | Why it matters |
|---|---|
| **Decode reads the whole KV cache** | Cost ∝ context length, memory-bound → PagedAttention, FP8 KV, GQA |
| **Prefill is n²** | Long prompts are expensive → FlashAttention (chunks + online softmax), chunked prefill |
| **√d** | Keeps softmax smooth. Engines pass it as `scale = 1/√d` |
| **Causal mask** | Lets prefill process all prompt tokens at once, in parallel |
| **Sliding window** | Look at only the last N tokens → cost stops growing (KV cache step 10) |

## Remember

- `softmax(QKᵀ / √d) × V` = **score → scale → mask → softmax → mix**.
- Q·K = a dot product (how well they match). V = the information that gets mixed.
- **÷ √d** stops softmax becoming all-or-nothing.
- Decode: Q is 1 row, cost = **4 × n × hidden** per layer, grows with context.
- Prefill: an **n × n** score matrix → FlashAttention never stores it.

## Quick check

1. What are the 5 steps of attention, in order?
2. Head size 64: what number do you divide the scores by?
3. In decode, what's the shape of the score row for a 1,000-token context?
4. Why does attention get slower as the chat gets longer, while the FFN doesn't?
5. Why doesn't FlashAttention store the n × n score matrix?

<details>
<summary>Answers</summary>

1. Score (QKᵀ) → scale (÷ √d) → mask → softmax → mix (× V).
2. √64 = **8**
3. **[1 × 1,000]**
4. Attention reads **every old token's K and V**. The FFN only processes the **new** token.
5. It's **n²** in size, too big. FlashAttention works in chunks with online softmax (3.2), so it never needs the full matrix.

</details>
