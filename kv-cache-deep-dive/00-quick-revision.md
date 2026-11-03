# ⚡ KV Cache — Quick Revision (read in 5 minutes)

> One page. Everything from steps 1–20. Read top to bottom before an interview or a new lesson.

---

## 1. Basics (steps 1–4)

| Idea | One line |
|---|---|
| How an LLM writes | One word at a time. Each new word looks back at **all** old words. |
| Q, K, V | **Q** = question ("what am I looking for?"), **K** = label ("what do I have?"), **V** = content ("here is my info"). |
| Without cache | Remake K,V for every old word, every round. 1,000 words → **500,500** K,V makes. |
| With cache | Make K,V once per word, save it. 1,000 words → **1,000** K,V makes. |
| Why save K,V but not Q | Only the **newest** word asks a question. Old words' Q is never used again. |

🧠 Analogy: K,V = notes you wrote in class. Don't rewrite them before every exam, just reread them.

---

## 2. Core (steps 5–7)

| | Prefill | Decode |
|---|---|---|
| What | Read the whole prompt at once | Write answer words one by one |
| Limited by | **Compute** (maths) | **Memory** (reading K,V + weights) |
| Sets | **TTFT** (time to first word) | **Words per second** |

**KV cache size formula**

```
KV size = 2 × layers × KV heads × head size × bytes × words × users
          ↑ K and V
```

- Llama 3 8B → **128 KB per word** → **1 GB per user** at 8K words
- GPU memory needed = **weights + KV cache + ~10%**
- H100 (80 GB) → about **56 users** (FP16) or **~112 users** (FP8)

**Why decode is slow:** reading memory ≈ **5 ms**, maths ≈ **0.016 ms**. The GPU mostly waits.
**Fix:** batching. Read the weights once, serve many users.

---

## 3. Making it smaller (steps 8–10)

| Trick | What it does | Saving |
|---|---|---|
| **MHA** | Every Q head has its own K,V head | none (baseline) |
| **GQA** | Q heads share K,V heads (Llama 3: 32 Q → **8 KV**) | **4× smaller** |
| **MQA** | All Q heads share **1** K,V head | biggest, but quality drops |
| **FP8 KV** | 2 bytes → 1 byte per number (`--kv-cache-dtype fp8`) | **2× smaller** |
| **Sliding window** | Keep only the last N words, **delete** older ones (not summarise) | Mistral: 4,096. Gemma 3: 5 window : 1 full layers |

---

## 4. Managing it on a server (steps 11–14)

| Idea | One line |
|---|---|
| **Fragmentation** | Old way reserved space for the max length + left gaps → **60–80% wasted**. |
| **PagedAttention** | Give small pages **only when needed**, place them **anywhere**, track them with a **page table**. Waste **< 4%**. vLLM: `--block-size 16`. |
| **Continuous batching** | A new user takes a seat **as soon as** one finishes. No waiting for the whole batch. |
| **Prefix caching** | Same starting text (system prompt) → reuse its K,V. Reuses the **question's** K,V, **not the answer**. Put fixed text **first**. `--enable-prefix-caching`. |

🧠 Analogy: PagedAttention = a library giving you one shelf at a time, anywhere in the building, with an index card telling you where.

---

## 5. Advanced (steps 15–17)

| Idea | One line |
|---|---|
| **Offloading** | Move unused K,V to CPU RAM (500 GB–2 TB, slower). Copy back when needed. |
| **Eviction (LRU)** | Memory full → throw out notes **used longest ago**. Server may **pause** a user. |
| **Disaggregation** | **Prefill GPUs** make K,V → send to **decode GPUs**. Smooth answers, but needs a fast link. |

---

## 6. Build it in code (steps 18–20)

| File | Shows |
|---|---|
| `code/01_attention_no_cache.py` | K,V made = **n(n+1)/2** → 10 words = **55** |
| `code/02_attention_with_cache.py` | Same answer. Speed-up **2×** at d=64, **24×** at d=512 |
| `code/03_paged_kv_cache.py` | Pages given one at a time, stored anywhere, page table, reused when freed |

Why small speed-up at d=64? Making K,V is cheap at small size, and **reading** all K,V still happens every round. The cache saves **making**, not **reading**.

---

## 🔢 Numbers to remember

| Number | Meaning |
|---|---|
| 500,500 vs 1,000 | K,V makes for 1,000 words: no cache vs cache |
| n(n+1)/2 | K,V makes without cache |
| 128 KB / word | Llama 3 8B KV size (FP16) |
| 1 GB / user | at 8K words |
| 56 → 112 | users on one H100, FP16 → FP8 |
| 32 → 8 | Llama 3 Q heads → KV heads (4× smaller) |
| 60–80% → < 4% | memory waste: old way → PagedAttention |
| 5 ms vs 0.016 ms | decode: memory read vs maths |

## ⚙️ vLLM flags

```
--kv-cache-dtype fp8        # half-size KV cache
--block-size 16             # 16 words per page
--enable-prefix-caching     # reuse shared prompt K,V
```

---

## 🚫 Common traps

| Wrong | Right |
|---|---|
| KV cache stores text | It stores **K,V numbers**. The app keeps the text. |
| KV cache stores only the prompt | It stores **every word so far**, answer words too. |
| We also cache Q | No. Only the newest word's Q is ever used. |
| Prefix caching returns a saved answer | It reuses the **prompt's K,V**. The answer is still generated. |
| Sliding window summarises old words | It **deletes** them. |
| Cache makes decode compute-bound | Decode is still **memory-bound**. The cache saves making K,V, not reading it. |
| One page holds a whole long chat | One page = **16 words** (vLLM). A long chat uses many pages. |

---

## ✅ 30-second self test

<details><summary>1. Why do we save K and V but not Q?</summary>Only the newest word asks a question. Old Qs are never needed again.</details>
<details><summary>2. Llama 3 8B, 8K words, 10 users — KV cache size?</summary>1 GB × 10 = about 10 GB.</details>
<details><summary>3. Prefill or decode: which is memory-bound?</summary>Decode.</details>
<details><summary>4. Llama 3 uses GQA (4× smaller). Add FP8 — how much smaller than plain MHA + FP16?</summary>4 × 2 = 8× smaller.</details>
<details><summary>5. What problem does PagedAttention solve?</summary>Wasted memory (fragmentation): 60–80% → under 4%.</details>
