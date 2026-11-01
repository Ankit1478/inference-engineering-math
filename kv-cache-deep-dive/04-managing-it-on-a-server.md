# KV Cache 4: Managing It on a Real Server

**A real server has many users at once. This lesson covers 4 ideas: the waste problem, PagedAttention (fixes the waste), continuous batching (no empty seats) and prefix caching (reuse K,V of the same start).**

## Quick summary (read this first)

**The problem (fragmentation)**

1. Nobody knows **how long an answer will be**.
2. Old systems **reserved memory for the maximum** (say 2,000 words) for every user, up front.
3. Most users need much less, so **most of the reserved memory sits empty**.
4. Users finish at different times and leave **gaps**. A new user needs one block **in a row**, so they may not fit even when the total free space is enough. This is called **fragmentation**.
5. The vLLM team found older systems **wasted 60–80%** of KV cache memory.

**The fix (PagedAttention)**

6. Split the cache into **small pages** (for example, 16 words each).
7. A user gets a **new page only when the last one is full**.
8. Pages can sit **anywhere** in memory. A **page table** remembers where each user's pages are.
9. Waste drops to **under 4%**, so **2–4× more users** fit on the same GPU.

**Continuous batching**

10. Old way: a group starts together and **waits for the slowest user** to finish. Seats sit empty.
11. New way: **as soon as one user finishes, the next one takes their seat**, at every decode round.

**Prefix caching**

12. Many messages **start the same way** (for example, the same system prompt).
13. Make K,V for that start **once**, then **reuse** it for every user. The first word comes **faster**.
14. It saves the **K,V of the question**, not the **answer**. The answer is still written for every user.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "If there's enough free memory in total, a new user fits" | Old systems needed one **block in a row**. Scattered gaps don't help. |
| "Reserving for the max is safe, so it's fine" | It's safe, but most of that memory is **never used**. |
| "PagedAttention makes the cache smaller" | Each word's K,V is the **same size**. It just **stops wasting** memory around it. |
| "A user's pages must sit next to each other" | Pages can be **anywhere**. The page table finds them. |
| "With pages there's zero waste" | Only the **last page** can be partly empty, so the waste is tiny, not zero. |
| "40 words fit in 1 page of 16" | 40 ÷ 16 = 2.5, round **up**: **3 pages**. |
| "A batch must wait until everyone finishes" | With continuous batching, a new user joins **as soon as a seat is free**. |
| "Prefix caching saves the answer" | It saves the **K,V of the question**. The answer is still **written again** for every user. |
| "Same question 2 times = answer written 1 time" | With only prefix caching, it's written **2 times**. Saving answers is **answer caching**, done by **your app**. |
| "A small change at the start is fine" | The start must be **exactly the same**. Put the **fixed text first** and the **changing text last**. |

## Why this matters for inference

- Memory decides **how many users fit on one GPU**. Wasting 60–80% means serving **2–4× fewer users**.
- Empty seats in a batch mean the GPU is **paid for but idle**.
- Re-reading the same system prompt for every user makes everyone **wait longer** for the first word.
- These 3 fixes are what made **vLLM** famous. Today almost every serving framework uses them.

## Real-life example

### The problem: the restaurant 🍽️

A restaurant gives **every group a 10-seat table**, "just in case friends join". Most groups are 2 people, so 8 seats sit empty at every table. The restaurant looks full, but it's mostly empty chairs.

### The problem: the parking lot 🚗

There are 3 free spots, but they're far apart. A bus needs 3 spots **next to each other**, so it can't park.

### PagedAttention: a notebook with loose pages 📒

Don't buy a 500-page notebook for every student "just in case". Give each student **loose pages**, one at a time, when they need one. The pages can be kept **anywhere** in the cupboard, and a **table of contents** says where each student's pages are.

| Notebook | Model |
|---|---|
| Loose page | A **block** of KV cache (for example, 16 words) |
| Table of contents | The **page table** (block table) |
| Cupboard | GPU memory |

### Continuous batching: a barber shop 💈

A barber shop has 3 chairs. When **one** haircut is done, the next customer sits down right away. The barber doesn't wait for all 3 haircuts to finish.

### Prefix caching: a photocopy 📄

A teacher hands out the same **instructions page** to 30 students. She writes it **once** and photocopies it. Each student only writes **their own answers**.

### Prefix caching vs answer caching: the chef 👨‍🍳

- **Prefix caching** = the vegetables are **already cut** 🔪. The chef still **cooks a fresh plate** for each customer.
- **Answer caching** = plates kept **ready in the fridge** 🍱. Nothing is cooked again.

## The math

### 1. Problem: reserving for the max

```
User A reserves: [■■■■□□□□□□□□□□□□□□□□]   used 200,   wasted 1,800 ❌
User B reserves: [■■□□□□□□□□□□□□□□□□□□]   used 100,   wasted 1,900 ❌
User C reserves: [■■■■■■■■■■■■□□□□□□□□]   used 1,200, wasted 800   ❌
```

```
Wasted % = (reserved − used) ÷ reserved × 100
```

### 2. Problem: gaps (fragmentation)

```
Memory: [A ■■■][  gap  ][C ■■■■][ gap ][E ■■]
```

The total free space is enough, but it's split into small pieces, so a new user who needs one big block **doesn't fit**.

### 3. The fix: PagedAttention

```
Memory:  [A1][B1][A2][C1][B2][A3][  ][  ]

User A's page table: page 1 → spot 1, page 2 → spot 3, page 3 → spot 6
```

User A's pages are **not next to each other**, and that's fine.

```
Pages needed = words ÷ words per page, rounded up
```

Page size 16, user has 40 words:

```
Page 1: words 1–16    ■■■■■■■■■■■■■■■■  full
Page 2: words 17–32   ■■■■■■■■■■■■■■■■  full
Page 3: words 33–40   ■■■■■■■■□□□□□□□□  8 used

40 ÷ 16 = 2.5 → 3 pages
3 × 16 = 48 spots → 48 − 40 = 8 empty spots (only in the last page)
```

| | Old way | PagedAttention |
|---|---|---|
| Memory given | the **max**, up front | **one page at a time** |
| Must be in a row? | yes | **no**, anywhere |
| Waste | **60–80%** | **under 4%** (only the last page) |

### 4. Continuous batching

A server with **room for 3 users**:

| User | Answer length |
|---|---|
| Riya | 5 words |
| Aman | 100 words |
| Neha | 5 words |
| Raj *(arrives a little later)* | 20 words |

**Old way: static batching**

```
Round 1–5:     [Riya][Aman][Neha]    Riya and Neha done ✅
Round 6–100:   [ 💤 ][Aman][ 💤 ]    2 seats empty, Raj waiting 😤
Round 101–120: [Raj ]                Raj finally starts
```

**New way: continuous batching**

```
Round 1–5:   [Riya][Aman][Neha]   Riya and Neha done ✅
Round 6:     [Raj ][Aman][    ]   Raj takes a free seat right away 🎉
Round 25:    Raj done ✅
Round 100:   Aman done ✅
```

| | Static | Continuous |
|---|---|---|
| Raj starts at | round **101** | round **6** |
| Everyone done at | round **120** | round **100** |
| Empty seats | many | almost none |

It needs PagedAttention: users join and leave **any time**, so pages must be given out and freed **any time**.

### 5. Prefix caching

```
User 1: [system prompt, 1,000 words] + "What is AI?"     → make K,V for 1,000 + 3
User 2: [system prompt, 1,000 words] + "Tell me a joke"  → reuse 1,000 ♻️ + make 3
User 3: [system prompt, 1,000 words] + "Hi"              → reuse 1,000 ♻️ + make 1
```

With pages, many users' page tables **point to the same pages**, so there's only 1 copy in memory:

```
Riya's page table:  [shared 1][shared 2][Riya's page]
Aman's page table:  [shared 1][shared 2][Aman's page]
```

| | Without | With prefix caching |
|---|---|---|
| Prefill work for the shared start | every user | **once** |
| TTFT (wait for the first word) | slow | **much faster** ⚡ |
| Memory for the shared start | 1 copy per user | **1 copy** total |

**What it saves and what it doesn't**

| | Reading the question (prefill) | Writing the answer (decode) |
|---|---|---|
| No caching | every user | every user |
| Prefix caching (server) | **once** ♻️ | every user |
| Answer caching (your app, optional) | once | **once** 🍱 |

**Where things are saved**

```
📱 User → 🧑‍💻 Your app (answer cache, optional, e.g. Redis) → 🖥️ Server (GPU: weights + KV cache)
```

The KV cache lives in the **server's GPU memory**. It's small and shared, so old unused notes are removed when it fills up (eviction, in the next lesson).

## Inference connection

```
PagedAttention      → no wasted memory     → more users per GPU
Continuous batching → no empty seats       → more words per second (2–20× in real tests)
Prefix caching      → no repeated prefill  → faster first word
```

| vLLM setting | What it does |
|---|---|
| `--block-size 16` | words per page (16 is the default) |
| `--enable-prefix-caching` | reuse K,V of the same start (on by default in newer versions) |

Continuous batching is always on in vLLM.

**Prompt tip:** put the **fixed text first** (system prompt, rules, documents) and the **changing text last** (the user's question, today's date).

## Remember

1. Old systems reserved for the **max** and left **gaps**: 60–80% wasted.
2. **PagedAttention** = small pages, given **only when needed**, stored **anywhere**, found with a **page table**. Waste under 4%.
3. **Continuous batching** = when one user finishes, the next takes their seat **right away**.
4. **Prefix caching** = make K,V for the same start **once** and reuse it. Faster first word.
5. Prefix caching saves **K,V of the question**, not the **answer**. Answer caching is the **app's** job.
6. The KV cache lives in the **server's GPU memory**.

## Quick check

1. A system reserves 1,000 words for every user. A chat ends after 250 words. What percentage was wasted?
2. There are 3 free gaps of 100 words each (300 total). A new user needs 200 words in a row. Do they fit in an old system?
3. Each page holds 16 words, and a user has written 40 words. How many pages do they need, and how many empty spots are in the last page?
4. Each page holds 16 words, and a user has written 50 words. How many pages?
5. In a batch of 4 users, 1 user finishes. With continuous batching, what happens to their seat?
6. With prefix caching only, 10 users ask the same question. How many times is the answer written?
7. Your prompt is `"Today is 1 Nov 2026. " + [1,000-word system prompt] + question`. Will prefix caching work well across days?

<details>
<summary>Answers</summary>

1. (1,000 − 250) ÷ 1,000 = **75%**
2. **No.** Each gap is only 100 words, and old systems need one block in a row.
3. 40 ÷ 16 = 2.5, so **3 pages**. 48 − 40 = **8 empty spots**.
4. 50 ÷ 16 = 3.1, so **4 pages**.
5. A **new user takes it right away**, in the next round.
6. **10 times.** Prefix caching only skips reading the question.
7. **Not well.** The date is at the start, so the start changes every day and nothing after it can be reused. Put the date **after** the system prompt.

</details>
