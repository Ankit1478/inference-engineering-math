# KV Cache 4: Managing It on a Real Server

**On a real server with many users, most KV cache memory used to be wasted. This lesson explains the waste. PagedAttention, continuous batching and prefix caching (coming next) fix it.**

## Quick summary (read this first)

1. Nobody knows **how long an answer will be**.
2. Old systems **reserved memory for the maximum** (say 2,000 words) for every user, up front.
3. Most users need much less, so **most of the reserved memory sits empty**.
4. Users finish at different times and leave **gaps**. A new user needs one block **in a row**, so they may not fit even when the total free space is enough. This is called **fragmentation**.
5. The vLLM team found older systems **wasted 60–80%** of KV cache memory.
6. Wasted memory means fewer users per GPU, so a **higher cost**.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "If there's enough free memory in total, a new user fits" | Old systems needed one **block in a row**. Scattered gaps don't help. |
| "Reserving for the max is safe, so it's fine" | It's safe, but most of that memory is **never used**. |

## Why this matters for inference

- Memory decides **how many users fit on one GPU**.
- Wasting 60–80% of the cache means serving **2–4× fewer users** than you could.
- Fixing this waste is what made **vLLM** famous.

## Real-life example

### The restaurant 🍽️

A restaurant gives **every group a 10-seat table**, "just in case friends join". Most groups are 2 people, so 8 seats sit empty at every table. The restaurant looks full, but it's mostly empty chairs.

### The parking lot 🚗

There are 3 free spots, but they're far apart. A bus needs 3 spots **next to each other**, so it can't park.

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

## Inference connection

```
Reserve for the max + gaps between users = most cache memory wasted (60–80%)
```

**The fix is coming next:** **PagedAttention** (vLLM) stores the cache in **small pages** that can sit **anywhere** in memory, and gives out a new page only when it's needed. That means no reserving for the max and no gaps problem.

## Remember

1. Answer length is **unknown**, so old systems reserved for the **max**.
2. Most reserved memory sat **empty**.
3. Users leave at different times, creating **gaps** (fragmentation).
4. 60–80% of the cache was wasted, so fewer users and a higher cost.

## Quick check

1. A system reserves 1,000 words for every user. A chat ends after 250 words. What percentage was wasted?
2. There are 3 free gaps of 100 words each (300 total). A new user needs 200 words in a row. Do they fit in an old system?

<details>
<summary>Answers</summary>

1. (1,000 − 250) ÷ 1,000 = **75%**
2. **No.** Each gap is only 100 words, and old systems need one block in a row.

</details>
