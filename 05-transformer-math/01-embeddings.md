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

