# 6.1 Big-O

**Big-O describes how the work grows when the input grows. It ignores exact numbers and keeps only the shape: flat, straight line or curve.**

## Quick summary (read this first)

1. Big-O answers: **"if the input doubles, what happens to the work?"**
2. **O(1)** = stays the same. **O(n)** = doubles. **O(n²)** = **4×**.
3. It **ignores constants**: 5n and 1,000n are both O(n).
4. For inference, **n = number of tokens** (context length).
5. **Key facts:**
   - Decode attention per token = **O(n)**: it reads all the old K,V.
   - Prefill attention = **O(n²)**.
   - FFN per token = **O(1)** in context length.
   - KV cache memory = **O(n)** per user.
6. **Big-O is not everything:** a huge O(1) (the FFN) can beat a small O(n) (attention) until n gets large.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| O(n) means exactly n steps | It means **grows in proportion to** n. 5n is also O(n). |
| FlashAttention makes attention O(n) | Compute is **still O(n²)**. It cuts **memory** from O(n²) to O(n). |
| O(1) means fast | It means **doesn't grow**. It can still be big (the FFN is O(1) in n but ~70% of the work). |
| O(n²) is always worse than O(n) | Only once n is big enough. Constants matter for small n. |

## Why this matters for inference

- It tells you what breaks when users send **longer prompts**.
- Long context (32K, 128K, 1M) is a Big-O problem: n² gets huge fast.
- Many tricks exist only to change the Big-O: the KV cache, FlashAttention memory, sliding windows, PagedAttention lookups.

## Real-life example

### Handshakes at a party 🤝

| Task | Work when there are n guests | Big-O | 10 → 20 guests |
|---|---|---|---|
| Ring the doorbell once | 1 | O(1) | same |
| Greet each guest | n | O(n) | 2× work |
| Everyone shakes hands with everyone | about n² ÷ 2 | O(n²) | **4× work** |

Attention in prefill is the handshake party: **every token looks at every other token**.

## The math

### 1. The common shapes

| Big-O | Name | n = 1K | n = 8K | n = 128K | Double n → |
|---|---|---|---|---|---|
| O(1) | constant | 1 | 1 | 1 | same |
| O(log n) | log | 10 | 13 | 17 | +1 |
| O(n) | linear | 1K | 8K | 128K | 2× |
| O(n log n) | | 10K | 106K | 2.2M | a bit more than 2× |
| O(n²) | quadratic | 1M | 64M | **17 billion** | **4×** |

(log here is log₂ from 3.1: how many times you can halve n.)

### 2. Drop the constants and the small parts

```
3n² + 100n + 7   →   O(n²)
```

For big n, the n² part wins. Everything else is a rounding error.

### 3. Inference Big-O table ⭐

**n = tokens in the context**

| Operation | Big-O in n | Why |
|---|---|---|
| Embedding lookup | O(1) per token | Read one row (5.1) |
| FFN, per token | O(1) | Each token goes through it alone (5.4) |
| **Decode attention, per new token** | **O(n)** | Reads every old K and V (5.2) |
| **Prefill attention** | **O(n²)** | Every token vs every token |
| Prefill attention **memory**, normal | O(n²) | Stores the n × n score matrix |
| Prefill attention **memory**, FlashAttention | **O(n)** | Never stores the matrix (3.2) |
| KV cache memory | O(n) per user | One K,V per token |
| K,V made, **no cache**, writing n tokens | O(n²) | n(n+1)/2 (KV cache step 18) |
| K,V made, **with cache** | **O(n)** | One per token |
| Sliding window decode attention | O(window) = O(1) in n | Only the last N tokens |
| Page table lookup (PagedAttention) | O(1) | Direct index |

### 4. Total cost of writing a whole answer

Generating n tokens, where each decode step costs O(current length):

```
1 + 2 + 3 + … + n ≈ n² ÷ 2   →   O(n²) total attention work
```

So even **with** a KV cache, a very long answer gets slower and slower: each new token reads more K,V than the one before.

## Inference connection

### Constants matter: when does attention beat the FFN? ⭐

Big-O says O(n) attention will **eventually** beat O(1) weights. When, exactly? Llama 3 8B, **per layer, per decode token**:

```
Weights (attention projections + FFN): 218M params → 2 × 218M = 436M FLOPs   (O(1))
Attention scores and mix:              4 × n × 4,096 FLOPs                     (O(n))

Equal when: 4 × n × 4,096 = 436M  →  n ≈ 26,600 tokens
```

| Context | Attention FLOPs vs weight FLOPs |
|---|---|
| 2K | ~8% |
| 8K | ~30% |
| **~27K** | **equal** |
| 128K | ~5× more |

👉 Short chats: the **weights** dominate. Long context (RAG, agents, long documents): **attention** dominates.

### What each trick changes

| Trick | Changes |
|---|---|
| KV cache | K,V making: O(n²) → **O(n)** |
| FlashAttention | Attention **memory**: O(n²) → **O(n)** (compute stays O(n²)) |
| Sliding window | Decode attention: O(n) → **O(window)** |
| GQA, FP8 KV | Same Big-O, **smaller constant** (4×, 2×) |
| Mamba / linear attention | Decode: O(n) → **O(1)** per token (a different model design) |

👉 Many big speed-ups (GQA, FP8, fusion) **don't change Big-O at all**. They shrink the constant, and in practice that matters just as much.

## Remember

- Big-O = **how the work grows** with n. Ignore constants.
- O(1) same, O(n) 2×, **O(n²) 4×** when n doubles.
- Decode attention **O(n)** per token, prefill attention **O(n²)**, FFN **O(1)**, KV cache **O(n)**.
- FlashAttention fixes **memory** (n² → n), not compute.
- Constants matter: for Llama 3 8B, attention only beats the weights at **~27K tokens**.

## Quick check

1. Prompt length goes from 4K to 16K tokens. How much more prefill attention work?
2. Same change. How much bigger is the KV cache per user?
3. Does FlashAttention reduce the FLOPs of attention?
4. What's the Big-O of `7n² + 300n + 5000`?
5. A chat at 2K tokens is slow. Is attention the main cost for Llama 3 8B?

<details>
<summary>Answers</summary>

1. 4× longer → 4² = **16×**
2. **4×** (O(n))
3. **No.** Compute is still O(n²). It saves **memory** and memory trips.
4. **O(n²)**
5. **No.** At 2K, attention is only ~8% of the weights' FLOPs. The weights (mostly the FFN) dominate.

</details>
