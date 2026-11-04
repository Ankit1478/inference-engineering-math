# 5.1 Embeddings (word → vector)

**An embedding turns each token ID into a list of numbers by looking up one row in a big table. At the end of the model, a second big table (the lm_head) turns the vector back into a score for every word.**

## Quick summary (read this first)

1. **Tokenizer:** text → **token IDs** (whole numbers). In English, 1 token ≈ ¾ of a word.
2. **Embedding table:** one row per token, `[vocab size × hidden size]`. Llama 3 8B: **128,256 × 4,096**.
3. **Embedding = a lookup**, not maths. Token 9,906 → read row 9,906.
4. **lm_head** (the last layer) does the reverse: vector → a score (logit) for **every** token → softmax (3.2) → sampling (4.2).
5. The two tables together are about **1 billion** of Llama 3 8B's parameters (~13%).
6. **For inference:** the lookup is cheap, but the **lm_head** is a big matmul every token, and a **big vocabulary** makes sampling slower.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| The model reads words | It reads **token IDs**. "unbelievable" may be 3 tokens. |
| Embedding is a matrix multiply | It's a **lookup**: just read one row. |
| Tokens per second = words per second | 1 token ≈ ¾ word, so **100 tokens ≈ 75 words**. |
| The embedding table is the only vocab-sized part | The **lm_head** is the same size and is used every token. |

## Why this matters for inference

- Everything is counted in **tokens**: speed (tokens/sec), price (per million tokens), context length, KV cache size.
- The **lm_head** is one of the biggest single matmuls in the model, and it runs for every generated token.
- **Vocabulary size** decides the cost of the lm_head and of sampling (4.2).
- The embedding table is split across GPUs in tensor parallel (vLLM: `VocabParallelEmbedding`).

## Real-life example

### A dictionary with coordinates 📖

Imagine a dictionary where each word, instead of a definition, has a **row of 4,096 numbers** describing its meaning.

| Step | Real life | Model |
|---|---|---|
| 1 | Split the sentence into words | **Tokenizer** → IDs |
| 2 | Look up each word's row | **Embedding** lookup |
| 3 | Think about the meaning | 32 transformer layers |
| 4 | Compare your idea to every word in the dictionary and score each | **lm_head** → logits |
| 5 | Pick a word | softmax + sampling |

Words with similar meanings ("king", "queen") have rows that point in similar directions, so their **dot product** (1.1) is high.

## The math

### 1. Text → token IDs

```
"Hello world" → [9906, 1917]      (2 tokens)
```

Rule of thumb for English: **1 token ≈ 0.75 words**, so 1,000 words ≈ 1,333 tokens.

### 2. Token IDs → vectors (lookup)

```
Embedding table E:  [128,256 rows × 4,096 columns]

Token 9906 → E[9906] → a vector of 4,096 numbers
```

**Shapes:** a batch of 2 users with 10 tokens each:

```
IDs:     [2, 10]
Vectors: [2, 10, 4096]     ← the [batch, tokens, hidden] tensor from 1.6
```

### 3. Size of the tables

```
Embedding table = 128,256 × 4,096 = 525 million parameters
In BF16         = 525M × 2 bytes  = 1.05 GB
```

Llama 3 8B has a **separate** lm_head of the same size, so the two together are **~1.05 billion parameters (2.1 GB)**.

(Some small models **tie** them, using one table for both, to save memory.)

### 4. The lm_head at the end

```
Final vector [4,096] × lm_head [4,096 × 128,256] → logits [128,256]
```

One score per token in the vocabulary, then softmax → sampling.

**FLOPs per generated token:**

```
2 × 4,096 × 128,256 = 1.05 billion FLOPs
```

That's about **6%** of the whole model's ~16 GFLOPs per token (2 × 8B, from 1.3).

### 5. The whole forward pass (big picture)

```
text → tokenizer → IDs
     → embedding lookup
     → 32 × [ RMSNorm → attention (RoPE, KV cache) → add
              RMSNorm → FFN (SwiGLU)               → add ]
     → RMSNorm → lm_head → logits → softmax → sampling → next token
```

Lessons 5.2–5.4 cover the attention and FFN parts.

## Inference connection

| Point | Why it matters |
|---|---|
| **Everything is in tokens** | Throughput (tokens/sec), price per million tokens, `max_model_len`, KV cache per token |
| **Embedding = lookup** | Almost free: read one row (4,096 × 2 bytes = 8 KB) per token |
| **lm_head = big matmul** | ~1 GFLOP per token, and its 1 GB of weights is read every decode step |
| **Big vocabulary** | Llama 2 had 32K tokens and Llama 3 has 128K. Text uses fewer tokens, but the lm_head and sampling cost more |
| **Tensor parallel** | vLLM splits the vocab rows across GPUs (`VocabParallelEmbedding`, `ParallelLMHead`) |
| **Prefill vs decode** | In prefill, the lm_head only needs to run on the **last** token, because only that one picks the next word |

## Remember

- Tokenizer: text → IDs. **1 token ≈ ¾ word.**
- Embedding = **lookup** in a `[vocab × hidden]` table.
- lm_head = vector → a score for every token → softmax → sampling.
- Llama 3 8B: 128,256 × 4,096 → **525M** per table, ~**1B** for both.
- The lm_head costs **~1 GFLOP per token** and is read every decode step.

## Quick check

1. Is the embedding step a matmul or a lookup?
2. A user's prompt is 3,000 English words. Roughly how many tokens is that?
3. What does the lm_head output, and how long is that list for Llama 3?
4. Why does a bigger vocabulary make decode a bit slower?
5. During prefill of a 1,000-token prompt, for how many tokens must the lm_head run?

<details>
<summary>Answers</summary>

1. A **lookup** (read one row).
2. 3,000 ÷ 0.75 ≈ **4,000 tokens**.
3. **Logits**: one score per token, **128,256** of them.
4. The lm_head matmul and the sampling both grow with the vocabulary size.
5. **1**, the last token. Only it predicts the next word.

</details>
