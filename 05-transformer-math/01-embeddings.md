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

