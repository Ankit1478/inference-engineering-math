# KV Cache 4: Managing It on a Real Server

**On a real server with many users, most KV cache memory used to be wasted. This lesson explains the waste and how PagedAttention fixes it. Continuous batching and prefix caching come next.**

## Quick summary (read this first)

**The problem (fragmentation)**

1. Nobody knows **how long an answer will be**.
2. Old systems **reserved memory for the maximum** (say 2,000 words) for every user, up front.
3. Most users need much less, so **most of the reserved memory sits empty**.
4. Users finish at different times and leave **gaps**. A new user needs one block **in a row**, so they may not fit even when the total free space is enough. This is called **fragmentation**.
5. The vLLM team found older systems **wasted 60–80%** of KV cache memory.
6. Wasted memory means fewer users per GPU, so a **higher cost**.

**The fix (PagedAttention)**

7. Split the cache into **small pages** (for example, 16 words each).
8. A user gets a **new page only when the last one is full**.
9. Pages can sit **anywhere** in memory. A **page table** remembers where each user's pages are.
10. Waste drops to **under 4%**, so **2–4× more users** fit on the same GPU.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "If there's enough free memory in total, a new user fits" | Old systems needed one **block in a row**. Scattered gaps don't help. |
| "Reserving for the max is safe, so it's fine" | It's safe, but most of that memory is **never used**. |
| "PagedAttention makes the cache smaller" | Each word's K,V is the **same size**. It just **stops wasting** memory around it. |
| "A user's pages must sit next to each other" | Pages can be **anywhere**. The page table finds them. |
| "With pages there's zero waste" | Only the **last page** can be partly empty, so the waste is tiny, not zero. |

## Why this matters for inference

- Memory decides **how many users fit on one GPU**.
- Wasting 60–80% of the cache means serving **2–4× fewer users** than you could.
- Fixing this waste with PagedAttention is what made **vLLM** famous. Today almost every serving framework does it.

## Real-life example

### The problem: the restaurant 🍽️

A restaurant gives **every group a 10-seat table**, "just in case friends join". Most groups are 2 people, so 8 seats sit empty at every table. The restaurant looks full, but it's mostly empty chairs.

### The problem: the parking lot 🚗

There are 3 free spots, but they're far apart. A bus needs 3 spots **next to each other**, so it can't park.

### The fix: a notebook with loose pages 📒

Don't buy a 500-page notebook for every student "just in case". Give each student **loose pages**, one at a time, when they need one. The pages can be kept **anywhere** in the cupboard, and a **table of contents** says where each student's pages are.

| Notebook | Model |
|---|---|
| Loose page | A **block** of KV cache (for example, 16 words) |
| Table of contents | The **page table** (block table) |
| Cupboard | GPU memory |

## The math

### Problem 1: reserving for the max

```
User A reserves: [■■■■□□□□□□□□□□□□□□□□]   used 200,   wasted 1,800 ❌
User B reserves: [■■□□□□□□□□□□□□□□□□□□]   used 100,   wasted 1,900 ❌
User C reserves: [■■■■■■■■■■■■□□□□□□□□]   used 1,200, wasted 800   ❌
```

```
Wasted % = (reserved − used) ÷ reserved × 100
```

### Problem 2: gaps (fragmentation)

```
Memory: [A ■■■][  gap  ][C ■■■■][ gap ][E ■■]
```

The total free space is enough, but it's split into small pieces, so a new user who needs one big block **doesn't fit**.

### The fix: PagedAttention

```
Memory:  [A1][B1][A2][C1][B2][A3][  ][  ]

User A's page table: page 1 → spot 1, page 2 → spot 3, page 3 → spot 6
```

User A's pages are **not next to each other**, and that's fine.

**How many pages does a user need?**

```
Pages needed = words ÷ words per page, rounded up
```

Page size 16, user has 40 words:

```
40 ÷ 16 = 2.5 → 3 pages
3 pages hold 48 spots → 48 − 40 = 8 empty spots (only in the last page)
```

| | Old way | PagedAttention |
|---|---|---|
| Memory given | the **max**, up front | **one page at a time** |
| Must be in a row? | yes | **no**, anywhere |
| Waste | **60–80%** | **under 4%** (only the last page) |

## Inference connection

```
Old:   reserve the max + gaps between users = 60–80% wasted
Paged: small pages, given when needed, anywhere = under 4% wasted
```

- Same GPU, **2–4× more users**, so a much lower cost per user.
- In vLLM the page size is `--block-size` (16 by default).
- The name comes from **paging**, the trick your computer already uses for RAM.

## Remember

1. Answer length is **unknown**, so old systems reserved for the **max**.
2. Most reserved memory sat **empty**, and users leaving created **gaps** (fragmentation): 60–80% wasted.
3. **PagedAttention** = small pages, given **only when needed**, stored **anywhere**.
4. A **page table** says where each user's pages are.
5. Only the last page can be partly empty, so waste is **under 4%**.

## Quick check

1. A system reserves 1,000 words for every user. A chat ends after 250 words. What percentage was wasted?
2. There are 3 free gaps of 100 words each (300 total). A new user needs 200 words in a row. Do they fit in an old system?
3. Each page holds 16 words, and a user has written 40 words. How many pages do they need, and how many empty spots are in the last page?
4. Does PagedAttention make each word's K,V smaller?

<details>
<summary>Answers</summary>

1. (1,000 − 250) ÷ 1,000 = **75%**
2. **No.** Each gap is only 100 words, and old systems need one block in a row.
3. 40 ÷ 16 = 2.5, so **3 pages**. 48 − 40 = **8 empty spots**.
4. **No.** It stops the **waste around** the cache. To shrink K,V itself, use GQA or FP8 (lesson 3).

</details>
