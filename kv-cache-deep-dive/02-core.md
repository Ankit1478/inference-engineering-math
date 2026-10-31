# KV Cache 2: Core Ideas

**We save K and V (not Q), the cache is filled in prefill and grows in decode, its size has a simple formula, and decode is slow because of memory reading, not maths.**

## Quick summary (read this first)

1. **Save K and V, not Q.** Only the newest word asks a question (Q). Old words' K and V are used again by every future word.
2. **Prefill** = reading your whole question **at once**. It **fills** the cache and decides **TTFT** (the wait for the first word).
3. **Decode** = writing the answer **one word at a time**. It **adds 1 word** to the cache each round and decides the **typing speed**.
4. **Size formula:**
   ```
   KV cache = 2 × layers × KV heads × head size × bytes × words × users
   ```
5. **GPU memory needed** = weights + KV cache + about 10% extra.
6. **Decode is slow because of memory reading**, not maths. To write 1 word, the GPU reads the whole model plus every user's cache.
7. **Fix: batching.** Read the model once and write 1 word for many users. But each user's cache must still be read.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "We cache Q too" | Q is only needed for the **newest** word. Old Qs are never used again. |
| "Prefill and decode are the same" | Prefill does **all prompt words at once**. Decode does **1 word per round**. |
| "Decode is slow because of heavy maths" | Decode is slow because of **reading memory**. The maths is about 300× faster than the reading. |
| "A faster-maths GPU makes decode faster" | A **faster-memory** GPU does (for example, H200 vs H100). |

## Why this matters for inference

- The size formula tells you **how many users fit on one GPU**, which decides the cost.
- Prefill vs decode explains **TTFT** vs **words per second**, the two speeds users feel.
- "Decode is limited by memory" explains why **batching**, **smaller caches** and **faster-memory GPUs** matter so much.

## Real-life example

### Why not save Q: the classroom 🙋

A new student joins and asks a **question (Q)**. To answer it, they look at **everyone's name tags (K)** and **notes (V)**.

Tomorrow another new student asks **their own** new question. Yesterday's question is useless now, but everyone's name tags and notes are **still needed**. So keep K and V and throw away the old Q.

### Prefill vs decode: an exam 📝

- **Prefill** = **reading the question paper.** You read it all quickly.
- **Decode** = **writing the answer.** You write one word after another.

### Why decode is slow: the fast chef 👨‍🍳

A chef cooks a dish in **1 second**, but the walk to the shop for ingredients takes **5 minutes**. The walk decides everything.

- **GPU** = the chef (maths is fast)
- **Memory reading** = the walk (slow)
- **Batching** = one trip to the shop, then cook for 50 people

## The math

### 1. Why K and V, but not Q

When word 4 ("food") arrives:

```
Q:  only "food" asks a question               → needs Q of "food" only
K:  check labels of  I, love, Indian, food    → needs K of ALL words
V:  take content of  I, love, Indian, food    → needs V of ALL words
```

```
Q    = used once (newest word only)   → don't save
K, V = used by every future word      → save (KV cache)
```

That's why it's the **KV** cache, not the QKV cache.

### 2. Prefill vs decode

```
Prefill:  [What, is, the, capital, of, France, ?]  → all at the same time
          → makes K,V for all 7 → fills the cache → writes "The"

Decode 1: "The"      → writes "capital"   (adds its K,V)
Decode 2: "capital"  → writes "is"        (adds its K,V)
Decode 3: "is"       → writes "Paris"     (adds its K,V)
```

| | Prefill | Decode |
|---|---|---|
| Words per round | **all** prompt words | **1** word |
| Rounds | 1 | one per answer word |
| The GPU waits on | **maths** (compute-bound) | **memory reading** (memory-bound) |
| The user feels it as | **TTFT** | **words per second** |
| The KV cache | **filled** | **grows by 1 word** each round |

Prompt = 1,000 words, answer = 200 words → **1** prefill round + **200** decode rounds.

### 3. KV cache size formula

```
KV cache = 2 × layers × KV heads × head size × bytes × words × users
```

| Piece | Why it's there |
|---|---|
| **2** | We save **K and V** |
| **layers** | Every layer has its own K,V |
| **KV heads** | Each KV head keeps its own K,V |
| **head size** | How many numbers are in one K (or one V) |
| **bytes** | 2 for BF16, 1 for FP8 |
| **words** | One K,V for every word |
| **users** | Every user has their own cache |

The first 5 pieces are fixed for a model. They're all in the model's `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim`).

**Llama 3 8B** (32 layers, 8 KV heads, head size 128, BF16):

```
Per word:  2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB
1 user,   8,000 words:  128 KB × 8,000 ≈ 1 GB
50 users, 8,000 words:  ≈ 50 GB   (the model itself is only 16 GB!)
```

**Llama 3 70B** (80 layers, 8 KV heads, head size 128, BF16):

```
Per word:  2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 0.33 MB
```

### 4. How many users fit on one GPU?

```
GPU memory needed = weights + KV cache + about 10% extra
```

Llama 3 8B on one H100 (80 GB), 8,000 words per user:

```
Weights:          16 GB
Extra (~10%):      8 GB
Left for cache:   80 − 16 − 8 = 56 GB
Each user:         1 GB
→ about 56 users at the same time
```

Llama 3 70B in BF16, 10 users × 4,000 words:

```
Weights:   140 GB
KV cache:  0.33 MB × 4,000 × 10 ≈ 13 GB
Extra:     ≈ 15 GB
Total:     ≈ 168 GB → 3 H100s (2 only give 160 GB)
```

### 5. Why decode is slow

Llama 3 8B on an H100, writing 1 word:

```
Read:  16 GB ÷ 3,350 GB per second ≈ 5 ms       🐢
Maths: 16 billion operations ÷ ~1,000 trillion per second ≈ 0.016 ms  ⚡
```

**Reading is about 300× slower than the maths.** The GPU mostly waits for data.

**Batching** helps because the weights are shared:

```
1 user:    read 16 GB → 1 word
50 users:  read 16 GB → 50 words
```

But each user's cache is **their own** and must be read every round:

```
50 users × 1 GB cache + 16 GB weights = 66 GB to read per round
```

So a big KV cache causes **two problems**:
1. It **fills memory**, so fewer users fit.
2. It **slows every decode round**, because there's more to read.

## Inference connection

| Change | KV cache |
|---|---|
| 2× words | 2× bigger |
| 2× users | 2× bigger |
| BF16 → FP8 | **½ size** ✅ |
| Fewer KV heads (GQA) | **smaller** ✅ |

- vLLM does this same "how many users fit" calculation when it starts.
- **H200 vs H100**: the same maths speed, but faster memory (4.8 vs 3.35 TB/s) and more of it (141 vs 80 GB), so decode is faster.

## Remember

1. Save **K and V** (used by all future words). Don't save **Q** (only the newest word needs it).
2. **Prefill** = all prompt words at once, fills the cache, decides **TTFT**.
3. **Decode** = 1 word per round, grows the cache, decides **words per second**.
4. `KV cache = 2 × layers × KV heads × head size × bytes × words × users`
5. `GPU memory = weights + KV cache + about 10% extra`
6. Decode is limited by **memory reading**, not maths. Batching helps, and a smaller cache helps.

## Quick check

1. The model is on word 50. Which words' **Q** does it need? Which words' **K and V**?
2. A user pastes a 10,000-word document and asks for a 1-line summary. Which part takes most of the time: prefill or decode?
3. Llama 3 8B on an H100 has 56 GB left for the cache. Each user needs 1 GB in BF16. In FP8, about how many users fit?
4. To make decode faster, should you buy a GPU with faster **memory** or faster **maths**?

<details>
<summary>Answers</summary>

1. **Q** of word 50 only. **K and V** of all 50 words.
2. **Prefill.** 10,000 words to read, but only a few words to write.
3. FP8 halves the cache to 0.5 GB per user: 56 ÷ 0.5 = **about 112 users**.
4. **Faster memory.** Decode waits on reading, not maths.

</details>
