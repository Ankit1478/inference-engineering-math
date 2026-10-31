# KV Cache 1: Basics

**An LLM writes one word at a time. The KV cache saves each word's K and V once, so the model doesn't make them again for every new word.**

## Quick summary (read this first)

1. An LLM writes **one word at a time**, and each new word **looks back at all the words so far**.
2. In attention, the new word asks a **question (Q)**, checks the **labels (K)** of all words, and takes the **content (V)** of the useful ones.
3. The K and V of old words **never change**.
4. **Without a cache**, the model makes K and V for every word again in every round, which is slow.
5. **With a cache**, it makes K and V **once** for each word, saves them and reuses them. For 1,000 words that's about **500× less work**.
6. The catch: the saved notes take **memory**. The KV cache trades memory for speed.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "The model writes the whole answer at once" | It writes **one word at a time**. Each word needs the one before it. |
| "The cache saves the answer" | It saves the **K and V** of every word, which are the model's notes about each word. |
| "The cache is free" | It makes things **faster** but uses **memory**. |

(Note: the model really works with **tokens**, which are pieces of words. In these notes, think "token = word".)

## Why this matters for inference

- Every chatbot uses a KV cache. Without it, long answers would be far too slow.
- The KV cache is often the **biggest user of GPU memory** on a real server, bigger than the model itself.
- Almost every famous inference trick (GQA, FP8 KV cache, PagedAttention, prefix caching) exists to make the KV cache smaller or better managed.

## Real-life example

### Writing a story with a friend ✍️

You and a friend write a story, one word each turn. Before adding your word, you **re-read the whole story** so your word makes sense.

- The story has 5 words: you read 5.
- The story has 500 words: you read 500. 😩

The longer the story gets, the more re-reading there is for every new word.

### The library 📚

- **Q (Query)** = your **question**: "I need books about France"
- **K (Key)** = the **label** on each book
- **V (Value)** = what's **inside** each book

The labels and contents of old books never change, so you don't need to make them again. **Save them. That pile of saved labels and contents is the KV cache.**

## The math

### An LLM writes one word at a time

You ask: **"What is the capital of France?"**

```
Round 1:  reads "What is the capital of France?"   → writes "The"
Round 2:  reads "... France? The"                  → writes "capital"
Round 3:  reads "... France? The capital"          → writes "is"
Round 4:  reads "... France? The capital is"       → writes "Paris"
```

### Without a cache: make K and V again every round

Sentence: **"I love Indian food"**

```
Round 1: K,V for  I                          → 1 made
Round 2: K,V for  I, love                    → 2 made
Round 3: K,V for  I, love, Indian            → 3 made
Round 4: K,V for  I, love, Indian, food      → 4 made
                                    Total =   10
```

K and V for "I" are **exactly the same** every round. That's wasted work.

### With a cache: make K and V once, then save them

```
Round 1: make K,V for "I"       → save    [I]
Round 2: make K,V for "love"    → save    [I, love]
Round 3: make K,V for "Indian"  → save    [I, love, Indian]
Round 4: make K,V for "food"    → save    [I, love, Indian, food]
                                  Total = 4
```

Each round makes K and V for **only the new word** and reads the old ones from memory.

### With bigger numbers

| 1,000 words | K,V made |
|---|---|
| Without cache | 1 + 2 + … + 1,000 = **500,500** |
| With cache | **1,000** |

About **500× less work**. 🔥

## Inference connection

```
KV cache = trade memory for speed
```

- **Faster**: each new word only makes its own K and V.
- **Costs memory**: the cache grows with **every word** and **every user**.

Real example, Llama 3 70B in BF16: about **0.33 MB per word**.

```
100 users × 8,000 words × 0.33 MB ≈ 262 GB
```

The model itself is 140 GB, so **the cache is almost 2× bigger than the model**.

| Situation | Biggest memory user |
|---|---|
| 1 user, short chat | Weights |
| Many users, long chats (real servers) | **KV cache** ⭐ |

## Remember

1. An LLM writes **one word at a time** and looks back at all old words.
2. Q = question, K = label, V = content.
3. Old words' K and V **never change**, so **save them**.
4. With a cache, each round makes K and V for **only the new word**.
5. The KV cache **grows** with every word and every user.

## Quick check

1. The model has written 10 words. To write word 11, how many old words does it look back at?
2. Without a cache, how many K,V are made in total for a 5-word sentence?
3. With a cache, you've made 100 words and word 101 arrives. How many K,V do you **make**, and how many do you **read from the cache**?
4. What does the KV cache cost?

<details>
<summary>Answers</summary>

1. **10**, all of them
2. 1 + 2 + 3 + 4 + 5 = **15**
3. Make **1** (for the new word). Read **100** from the cache.
4. **Memory.** It saves time but uses GPU memory, and it grows with every word and every user.

</details>
