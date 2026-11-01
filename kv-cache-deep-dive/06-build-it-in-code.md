# KV Cache 6: Build It in Code

**Write attention in PyTorch, first without a cache (slow), then with a cache (fast), and compare. Finally, build a tiny paged KV cache.**

## Quick summary (read this first)

1. **Without a cache**, every round rebuilds K,V for **all** words so far: 1 + 2 + … + n.
2. **With a cache** (coming next), every round makes K,V for **only the new word**: n in total.
3. The code uses the same attention formula you know: `softmax(QKᵀ / √d) × V`.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Q is made for every word each round" | Only the **newest** word's Q is needed. |
| "10 words means 10 K,V made without a cache" | 1 + 2 + … + 10 = **55**. |

## Step 18: attention without a cache

Code: [code/01_attention_no_cache.py](code/01_attention_no_cache.py)

```python
for t in range(1, len(words) + 1):
    seen = words[:t]          # all words so far
    K = seen @ W_k            # ❌ make K for ALL words again
    V = seen @ W_v            # ❌ make V for ALL words again
    kv_made += t
    q = seen[-1] @ W_q        # Q only for the newest word
    out = attention(q, K, V)
```

| Part | Meaning |
|---|---|
| `W_q, W_k, W_v` | the model's weights that turn a word into Q, K and V |
| `attention(...)` | `softmax(QKᵀ / √d) × V` |
| `for t in ...` | one round per new word |
| `K = seen @ W_k` | **the waste**: rebuilds K,V for every old word, every round |
| `kv_made` | counts the work: 1 + 2 + … + 10 = **55** |

A real model would turn `out` into the next word. Here the words are already given, so we can focus on the K,V work.

### Run it

```bash
pip install torch
python3 kv-cache-deep-dive/code/01_attention_no_cache.py
```

```
K,V made: 55
```

```
K,V made without a cache = 1 + 2 + … + n = n × (n + 1) ÷ 2
```

## Coming next

- Step 19: attention **with** a cache, then compare the speed
- Step 20: a simple **paged** KV cache

## Quick check

1. Change `10` to `100` words. What will `kv_made` print?
2. In the loop, which line is the waste?

<details>
<summary>Answers</summary>

1. 100 × 101 ÷ 2 = **5,050**
2. `K = seen @ W_k` and `V = seen @ W_v`: they rebuild K,V for **all** old words every round.

</details>
