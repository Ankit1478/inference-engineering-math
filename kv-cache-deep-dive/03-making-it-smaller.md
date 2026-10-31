# KV Cache 3: Making It Smaller

**Three ways to shrink the KV cache: share K,V between heads (GQA), store it in 1 byte (FP8), and only keep the last N words (sliding window).**

## Quick summary (read this first)

1. **Head** = one way of looking at the sentence. Each **query head** asks its own kind of question.
2. **MHA**: every head has its own K,V. That's the biggest cache.
3. **GQA** ⭐: heads **share** K,V in groups. The questions stay many, but the K,V notebooks are few. Llama 3: 32 query heads, **8 KV heads**, so the cache is 4× smaller.
4. **MQA**: all heads share **one** K,V. It's the smallest cache, but quality drops a bit.
5. **FP8 KV cache**: 1 byte instead of 2, so **half the size**. In vLLM: `--kv-cache-dtype fp8`.
6. **Sliding window**: keep only the **last N words**. The cache **never grows**. Old words are **deleted, not summarised**.
7. Tricks **stack**: GQA (÷4) × FP8 (÷2) = **8× smaller**.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Each head asks about one word" | Each head asks **one kind of question** about **all** the words. |
| "GQA reduces the query heads" | GQA reduces the **KV heads**. All query heads stay. |
| "Heads sharing a notebook get the same answer" | They ask **different questions**, so they find **different answers** in the same notebook. |
| "Sliding window summarises old words" | Old words are just **deleted**. Summarising is a different trick done by the **app**. |
| "The window is 2–3 words" | Real windows are **thousands** of words (Mistral 7B: 4,096; Gemma 3: 1,024). |
| "I can switch on sliding window for any model" | It's decided **when the model is trained**. You can't add it to Llama 3. |
| "`--quantization fp8` also shrinks the cache" | That's the **weights**. The cache needs `--kv-cache-dtype fp8`. |

## Why this matters for inference

- A smaller cache means **more users per GPU**, so lower cost.
- A smaller cache also means **less to read** in every decode round, so faster words.

## Real-life example

### Heads = detectives 🕵️

The model reads **"Ravi gave Priya his book"** in many ways at the same time:

| Head | Question it asks |
|---|---|
| Head 1 | **Who** did it? → Ravi |
| Head 2 | **To whom?** → Priya |
| Head 3 | Who does **"his"** mean? → Ravi |

Each head is a detective looking for a different clue. Its question is its **Q**.

### GQA = students sharing textbooks 📚

**4 students** each have a **different question**.

```
MHA: everyone has their own book       GQA: 2 students share 1 book

Student 1 → 📕                          Student 1 ┐
Student 2 → 📕                          Student 2 ┘→ 📕
Student 3 → 📕                          Student 3 ┐
Student 4 → 📕                          Student 4 ┘→ 📕

4 books 🎒🎒🎒🎒                         2 books 🎒🎒 ✅
```

Students 1 and 2 read the **same book** but look for **different answers**: "Who is the king?" finds page 5, and "When was the war?" finds page 20.

| Classroom | Model |
|---|---|
| Student | Query head (Q) |
| Their question | What that head looks for |
| Book | K,V (what the KV cache saves) |
| Bag weight | GPU memory |

### Sliding window = a small whiteboard 🧑‍🏫

The board fits only **10 lines**. When line 11 comes, line 1 is erased. The board never gets bigger.

## The math

### 1. MHA vs GQA vs MQA

Example: **"The cat sat on the mat"**, with 4 query heads.

```
MHA (4 KV heads): every word saves 4 notebooks
  Q1 → notebook 1    Q2 → notebook 2    Q3 → notebook 3    Q4 → notebook 4
  Cache: 6 words × 4 = 24 notebooks

GQA (2 KV heads): every word saves 2 notebooks
  Q1, Q2 → notebook A
  Q3, Q4 → notebook B
  Cache: 6 words × 2 = 12 notebooks (half) ✅
```

| | Q heads | KV heads | Cache |
|---|---|---|---|
| MHA | 4 | 4 | 24 |
| **GQA** | 4 | **2** | **12** ✅ |
| MQA | 4 | 1 | 6 |

**Llama 3**: 32 query heads, groups of 4 share, so 32 ÷ 4 = **8 KV heads**.

```
MHA: 32 KV heads → 100%  → 512 KB per word (Llama 3 8B)
GQA:  8 KV heads →  25%  → 128 KB per word  ⭐
MQA:  1 KV head  →   3%
```

```
KV heads = query heads ÷ heads per group
```

### 2. FP8 KV cache

Llama 3 8B on an H100 (56 GB free for the cache, 8,000 words per user):

| KV cache in | Per word | Per user | Users that fit |
|---|---|---|---|
| BF16 | 128 KB | 1 GB | **56** |
| **FP8** | **64 KB** | **0.5 GB** | **112** 🚀 |

| vLLM setting | What it shrinks |
|---|---|
| `--quantization fp8` | the **weights** |
| `--kv-cache-dtype fp8` | the **KV cache** |

There's a little loss of detail, so **always test quality**, especially on long chats.

### 3. Sliding window

```
Window = 4 words

Words:   The  cat  sat  on  the  mat  and  slept
Cache:                       [the  mat  and  slept]
```

A new word comes in and the oldest falls out. **No summary, just deleted.**

The size is **fixed**, so the GPU **reserves the memory once** and overwrites the oldest spot, like a clock going from 12 back to 1:

```
Spots:          [1]  [2]  [3]  [4]
                 the  cat  sat  on
"mat" arrives →  mat  cat  sat  on
"and" arrives →  mat  and  sat  on
```

Chat = 32,000 words, window = 4,000 words → the cache holds **4,000**, so it's **8× smaller**.

**A trace of old words survives:** when "mat" was made, it looked at "cat", so a little of "cat" lives inside "mat". It's like a rumour passed along, not a clean summary.

**The catch:** the model can't directly see words older than the window. Real models **mix** layer types:

```
Layers 1–5: sliding window  🪟  (small cache)
Layer 6:    full            👁️  (sees the whole chat)
```

**Gemma 3** uses 5 window layers for every 1 full layer. **Mistral 7B** used a sliding window of 4,096.

| | Normal cache | Sliding window |
|---|---|---|
| Size | keeps growing 📈 | **fixed** ➖ |
| Memory | hard to plan | **reserve once** ✅ |
| Remembers | everything | only the last N words |

| | Sliding window | Summarising |
|---|---|---|
| Who does it | the **model** (built in) | the **app** |
| Old words | **deleted** | **turned into a short summary** |

## Inference connection

Tricks stack. Start from MHA in BF16:

```
GQA (4 share)  →  ÷4
FP8            →  ÷2
Together       →  8× smaller 🔥
```

| Trick | Who decides |
|---|---|
| GQA / MQA | the model's **design** (you can't change it) |
| Sliding window | the model's **design** (you can't change it) |
| FP8 KV cache | **you**, when serving (`--kv-cache-dtype fp8`) |

## Remember

1. A head = one way of asking about **all** the words. Query heads ask the questions.
2. **GQA** = many query heads share fewer KV heads, so a smaller cache with almost the same quality.
3. `KV heads = query heads ÷ heads per group`. Llama 3: 32 ÷ 4 = 8.
4. **FP8 KV cache** = half the size, so 2× the users.
5. **Sliding window** = keep the last N words, delete older ones, and the cache size is fixed.
6. GQA × FP8 = **8× smaller**.

## Quick check

1. A model has 64 query heads, and groups of 8 heads share. How many KV heads?
2. 8 students, every 4 share 1 book. How many books?
3. In GQA, what gets reduced: query heads or KV heads?
4. A user's cache is 2 GB in BF16. How big is it in FP8?
5. Window = 1,000 words, and the chat is 5,000 words. How many words are in the cache?
6. In a sliding window, is the oldest word summarised or deleted?

<details>
<summary>Answers</summary>

1. 64 ÷ 8 = **8**
2. 8 ÷ 4 = **2**
3. **KV heads**
4. **1 GB**
5. **1,000**
6. **Deleted**

</details>
