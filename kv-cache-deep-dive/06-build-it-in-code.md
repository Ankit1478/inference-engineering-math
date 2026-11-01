# KV Cache 6: Build It in Code

**Write attention in PyTorch, first without a cache (slow), then with a cache (fast), and compare. Finally, build a tiny paged KV cache (the PagedAttention idea).**

## Quick summary (read this first)

1. **Without a cache**, every round rebuilds K,V for **all** words so far: 1 + 2 + … + n.
2. **With a cache**, every round makes K,V for **only the new word**: n in total.
3. Both give the **same answer**. The cache only skips repeated work.
4. The speed-up is **bigger for bigger models**: 2× with tiny words (64 numbers), about **20×** with bigger words (512–1,024 numbers).
5. A **paged** KV cache gives pages **one at a time**, stores them **anywhere**, and **reuses** freed pages. It still gives the exact same K,V.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Q is made for every word each round" | Only the **newest** word's Q is needed. |
| "10 words means 10 K,V made without a cache" | 1 + 2 + … + 10 = **55**. |
| "The cache changes the answer a little" | Same answer (`Same answer? True`). Only the speed changes. |
| "1,000× less work means 1,000× faster" | Both versions still **read** all old K,V every round. The cache saves the **making**, not the **reading**. |
| "Paged pages must be next to each other" | Pages can be **anywhere**. The page table finds them, and the K,V is still the same. |

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

```
K,V made without a cache = 1 + 2 + … + n = n × (n + 1) ÷ 2
```

## Step 19: attention with a cache, then compare

Code: [code/02_attention_with_cache.py](code/02_attention_with_cache.py)

```python
K_cache = torch.zeros(n, d)         # the KV cache 📦
V_cache = torch.zeros(n, d)
for t in range(n):
    x = words[t]                    # only the NEW word
    K_cache[t] = x @ W_k            # ✅ make its K once, save it
    V_cache[t] = x @ W_v            # ✅ make its V once, save it
    q = x @ W_q
    out = attention(q, K_cache[:t + 1], V_cache[:t + 1])
```

| | Without cache | With cache |
|---|---|---|
| Each round makes K,V for | **all** words so far | **only the new word** |
| K,V made for 2,000 words | **2,001,000** | **2,000** |

### Real output (run on a Mac CPU)

**Same answer?**

```
Output for word 2,000 (first 4 numbers):
  Without cache: [10.4997, -5.7159, 6.2894, 2.5226]
  With cache:    [10.4997, -5.7159, 6.2894, 2.5226]
  Biggest difference anywhere: 4.3e-04   (tiny rounding noise)
  Same answer? True
```

**Speed, 64 numbers per word:**

| Words | Without cache | With cache | Faster by |
|---|---|---|---|
| 500 | 0.01 s | 0.01 s | 1.2× |
| 1,000 | 0.04 s | 0.02 s | 1.4× |
| 2,000 | 0.13 s | 0.06 s | 2.1× |
| 4,000 | 0.36 s | 0.17 s | 2.1× |

**Speed, 2,000 words, bigger words:**

| Numbers per word | Without cache | With cache | Faster by |
|---|---|---|---|
| 64 | 0.13 s | 0.06 s | **2×** |
| 512 | 2.73 s | 0.11 s | **24×** 🚀 |
| 1,024 | 9.32 s | 0.51 s | **18×** 🚀 |

(Your times will be different on your machine.)

### Why only 2× with small words, if the work is 1,000× less?

1. **Making K,V for a tiny word is cheap**, so skipping it saves little. With bigger words it gets expensive, so skipping it helps a lot.
2. **Both versions still read all old K,V every round** for attention. The cache saves the **making**, not the **reading** (lesson 2: decode is limited by **reading memory**).
3. Llama 3 8B has words of **4,096 numbers** and **32 layers**, so the cache saves far more than in this toy.

**Note:** `torch.zeros(n, d)` reserves space for **all** words up front. That's the "reserve for the max" problem from lesson 4. Fine for 1 user, wasteful for many. Step 20 fixes it.

## Step 20: a simple paged KV cache

Code: [code/03_paged_kv_cache.py](code/03_paged_kv_cache.py)

Setup: **6 pages** of memory, **4 words** per page.

```python
class PagedKVCache:
    def __init__(self):
        self.K = torch.zeros(NUM_PAGES, PAGE_SIZE, d)   # all GPU memory, cut into pages
        self.free_pages = list(range(NUM_PAGES))        # pages nobody is using
        self.page_table = {}                            # user → list of their pages

    def add_word(self, user, k, v):
        if n % PAGE_SIZE == 0:                          # last page full?
            pages.append(self.free_pages.pop(0))        # take a new page, from anywhere
        self.K[page, slot] = k                          # write into that page

    def get(self, user):
        K = self.K[pages].reshape(-1, d)[:n]            # follow the page table

    def finish(self, user):
        self.free_pages += self.page_table.pop(user)    # give the pages back
```

| Part | What it does | From which step |
|---|---|---|
| `free_pages` | the list of empty pages | 12 PagedAttention |
| `add_word` | takes a new page **only when the last one is full** | 12 |
| `page_table` | remembers **where** each user's pages are | 12 |
| `finish` | frees the pages when a user is done | 13 continuous batching |

### Real output

```
1. Riya and Aman take turns writing words:
  Memory: [Riya] [Aman] [Riya] [  --] [  --] [  --]
  Riya's page table: [0, 2]  (6 words)      ← not next to each other ✅
  Aman's page table: [1]  (3 words)

2. Riya finishes, her pages are freed:
  Memory: [  --] [Aman] [  --] [  --] [  --] [  --]

3. Raj arrives and writes 9 words (3 pages):
  Memory: [  --] [Aman] [  --] [ Raj] [ Raj] [ Raj]
   Raj's page table: [3, 4, 5]  (9 words)

4. Neha arrives, writes 5 words and gets Riya's old pages:
  Neha's pages: [0, 2] (not next to each other)
  Same? True                                 ← same K,V as a normal cache ✅
```

What it shows:

- Pages are given **one at a time**, so there's no reserving for the max.
- Pages can be **anywhere** (Riya got 0 and 2).
- Freed pages are **reused right away** (Neha got Riya's old pages).
- Scattered pages still give the **exact same K,V**.

Real vLLM does the same thing on the GPU, with special attention code that reads straight from the pages.

## Run it

```bash
pip install torch
python3 kv-cache-deep-dive/code/01_attention_no_cache.py
python3 kv-cache-deep-dive/code/02_attention_with_cache.py
python3 kv-cache-deep-dive/code/03_paged_kv_cache.py
```

## Remember

1. Without a cache: K,V made = n × (n + 1) ÷ 2. With a cache: **n**.
2. Same answer, less work. The **bigger the model**, the bigger the speed-up.
3. The cache saves **making** K,V, not **reading** it.
4. A paged cache = pages **one at a time**, **anywhere**, found with a **page table**, **reused** when freed.

## Quick check

1. Change `10` to `100` words in step 18. What will `kv_made` print?
2. In the step 18 loop, which line is the waste?
3. In step 19, did the cache change the answer, or only the speed?
4. With the cache, how many K,V are made for 2,000 words?
5. Riya wrote 6 words with 4 words per page. Why did she get 2 pages, not 1?

<details>
<summary>Answers</summary>

1. 100 × 101 ÷ 2 = **5,050**
2. `K = seen @ W_k` and `V = seen @ W_v`: they rebuild K,V for **all** old words every round.
3. **Only the speed.** `Same answer? True`.
4. **2,000**, one per word.
5. One page holds only 4 words. 6 ÷ 4 = 1.5, which rounds up to **2 pages**.

</details>
