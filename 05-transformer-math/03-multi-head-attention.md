# 5.3 Multi-head attention

**Instead of one big attention, the model runs many small attentions (heads) side by side, each looking for something different, then joins their results.**

## Quick summary (read this first)

1. **Split** the 4,096 numbers into **32 heads × 128** numbers each.
2. Each head runs the attention formula (5.2) **on its own**.
3. **Join** (concatenate) the 32 results back into 4,096, then multiply by the output matrix **W_O**.
4. Each head can learn a different job: grammar, the previous word, names, and so on.
5. **GQA:** many Q heads **share** fewer K,V heads. Llama 3 8B has **32 Q heads, 8 KV heads** → 4× smaller KV cache (KV cache step 8).
6. **For inference:** heads are **independent**, so they split perfectly across GPUs (tensor parallel), and GPU kernels run them in parallel.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| 32 heads = 32× the work | Each head is **1/32 the size**, so the total work is the same as one big head. |
| More heads = a bigger KV cache | The KV cache depends on the number of **KV heads**, not Q heads. |
| GQA removes Q heads | GQA keeps **all Q heads** and shares **K,V heads**. |
| Tensor parallel can use any number of GPUs | It splits by **heads**: with 8 KV heads, going above 8 GPUs means copying K,V heads. |

