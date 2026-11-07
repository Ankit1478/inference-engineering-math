# 6.5 Arithmetic intensity and the roofline model ⭐

**The roofline answers one question: is the GPU slow because of memory (reading weights) or because of maths? It all comes from `time = distance ÷ speed`. Example: Llama 3 8B on one H100.**

| Step | Idea |
|---|---|
| 1 | **Answers per read** (arithmetic intensity) |
| 2 | **The speed limit**: fill the bus (500 seats) |
| 3 | **Prefill vs decode**: where real jobs sit on the roof |
| 4 | **Fixing decode**: batching and smaller weights |
| 5 | **Fixing prefill**, and the KV cache that batching can't fix |

## Step 1: How many answers per read? 📖

**Story:** before answering, a student must **re-read the whole textbook** (slow). Each answer then takes only **1 second** (fast). So ask: **for one reading of the book, how many answers do I get?**

| Students asking at once | Book reads | Answers | Answers per read |
|---|---|---|---|
| 1 | 1 | 1 | **1** |
| 10 | 1 | 10 | **10** |
| 64 | 1 | 64 | **64** |

That number has a fancy name: **arithmetic intensity** = work done per read.

### Why the GPU reads all the weights for every word

- Llama 3 8B = **8 billion weights × 2 bytes = 16 GB**. This is the "textbook".
- The model is **32 layers**, like floors in a building. Every word must walk through **every floor** and get multiplied by that floor's weights. No floor can be skipped.
- So **every new word = one read of (almost) all 16 GB**. It doesn't matter if the message is "Hi" or a long question.
- The GPU's fast on-chip memory is only ~50 MB, so the 16 GB can't stay close. It's re-read from GPU memory for every word.

| Exception | What gets read |
|---|---|
| Embedding table (5.1) | only **1 row** per word, so really ~15 GB, not 16 |
| MoE models (5.4) | only the **experts the router picks** |

### When you send "Hi" (ChatGPT, Claude and local models all work this way)

| When | Reads of the weights |
|---|---|
| Your message (prefill) | **1 read** for all its tokens together |
| The reply (decode) | **1 read for every word** |

Example: you send "Hi" and the reply is 5 words, so that's **1 + 5 = 6 reads**. Big services make this fast and cheap by splitting the weights over many GPUs, using smaller weights (FP8/INT4, MoE), and **sharing each read among thousands of users**.

### On the GPU: users share one read

Every user's next word needs the **same** weights. So the GPU reads 16 GB **once** and uses it for all users:

| | Reads of 16 GB | Time | Words made |
|---|---|---|---|
| 1 user | 1 | 8 ms | **1** |
| 64 users | 1 | ~8 ms | **64** |

👉 **Easy rule: intensity ≈ how many users (tokens) share one read of the weights.**

### The problem this shows

| Job (1 user, 1 word) | Time |
|---|---|
| Read 16 GB of weights | **8 ms** 🐢 |
| Do the maths | **0.016 ms** ⚡ |

The maths unit is idle **99.8%** of the time. An expensive GPU waits on memory. **The fix: more answers per read.**

**Check:** 200 users at once. Intensity? → **~200** (one read gives 200 words)

## Step 2: The speed limit 🚌

### The bus story

- A bus trip **always takes 8 minutes**, empty or full.
- The bus has **500 seats**.

| People waiting | People moved per trip |
|---|---|
| 1 | **1** (499 seats empty 😴) |
| 100 | **100** |
| 500 | **500** (full ✅) |
| 1,000 | still **500** (the rest wait) |

The line **goes up, then goes flat**, like a roof. That's the **roofline**. 🏠

| Bus | GPU |
|---|---|
| A trip (8 minutes) | Reading the weights (8 ms) |
| Seats (500) | Words whose maths fits in one read |
| People | Users (each wants 1 word) |

### The maths: `time = distance ÷ speed`, used twice

| | "Distance" (work) | Speed | Time |
|---|---|---|---|
| 🚚 Reading | 16 GB of weights | 2,000 GB/s | **8 ms** |
| 👨‍🍳 Maths (1 word) | 2 × 8B = 16 billion steps | 1,000 trillion/s (1,000 TFLOPS) | **0.016 ms** |

```
1. Trip time   = weights ÷ memory speed        = 16 ÷ 2,000      = 8 ms
2. Word maths  = (2 × weights) ÷ maths speed   = 16 G ÷ 1,000 T  = 0.016 ms
3. Seats       = trip time ÷ word maths        = 8 ÷ 0.016       = 500
4. GPU busy    = users ÷ seats
```

The **500** is the same ratio as in 6.4 (GPU FLOPS ÷ bandwidth).

### Many users: the slower job decides the time

- **Reading time stays 8 ms**, because the weights are read **once** and shared by all users.
- **Maths time grows**, because each user needs their own maths: `maths time = users × 0.016 ms`.
- **Real time = the bigger of the two**, like waiting for the last friend before leaving.

| Users | Reading time | Maths time | **Real time** | Slow part |
|---|---|---|---|---|
| 1 | 8 ms | 0.016 ms | **8 ms** | 🚚 reading |
| 64 | 8 ms | 1 ms | **8 ms** | 🚚 reading |
| 250 | 8 ms | 4 ms | **8 ms** | 🚚 reading |
| **500** | 8 ms | **8 ms** | **8 ms** | equal ⚖️ |
| 1,000 | 8 ms | 16 ms | **16 ms** | 👨‍🍳 maths |

- **Up to 500 users:** the time stays 8 ms, so extra users are **free**. 🎉
- **After 500:** the maths takes longer, so time goes up. No more free users.

The switch point is where both times are **equal**:

```
users × 0.016 = 8   →   users = 8 ÷ 0.016 = 500
```

### Fill the seats

| Users | GPU busy | Words per second (125 trips/s × users) | What's slow |
|---|---|---|---|
| 1 | **0.2%** | 125 | 🚚 memory |
| 64 | **13%** | 8,000 | 🚚 memory |
| 250 | **50%** | 31,250 | 🚚 memory |
| **500** | **100%** | **62,500** | balanced ⚖️ |
| 1,000 | 100% | 62,500 (no more!) | 👨‍🍳 maths |

```
words/sec
62,500 |            _______________  ← flat roof (maths limit)
       |          /
       |        /
       |      /    ← slope (memory limit)
       |    /
       |  /
       |/________________________ users per read
                  500
```

### The same thing in TFLOPS (how engineers write it)

```
Speed = the smaller of:  2 × intensity   (memory: 2,000 GB/s × intensity)
                         1,000           (maths: the GPU's max)
```

| Intensity | Memory limit | Maths limit | Real speed |
|---|---|---|---|
| 1 | 2 | 1,000 | **2 TFLOPS** |
| 64 | 128 | 1,000 | **128 TFLOPS** |
| 500 | 1,000 | 1,000 | **1,000 TFLOPS** |
| 2,000 | 4,000 | 1,000 | **1,000 TFLOPS** |

👉 **Below 500: the slope, memory is the limit, so add users. Above 500: the flat roof, maths is the limit.**

**Check:** 100 users. GPU busy? What's slow? → 100 ÷ 500 = **20%**, memory (the slope)

## Step 3: Prefill vs decode (where real jobs sit) 🏠

The only question: **how many tokens share one read of the weights?**

| Job | What it is | Tokens per read |
|---|---|---|
| **Prefill** | reading the prompt: all its tokens go through **together** | = prompt length |
| **Decode** | writing the reply: **1 word per user** per read | = number of users |

- In decode, **every word reads all 16 GB again**. The weights are only **read, never updated**.
- A 2,000-token prompt = **1 read**. A 2,000-word reply = **2,000 reads**.

### FLOPs vs bytes

**FLOP** = 1 maths step (1 multiply or 1 add). Each weight = **2 FLOPs per token**. **Bytes** = what the truck brings, **FLOPs** = what the chef cooks.

| Job | Bytes read | FLOPs | Reading | Maths | Slow part | FLOPs per byte |
|---|---|---|---|---|---|---|
| Decode, 1 user | 16 GB | 16 GFLOPs | **8 ms** | 0.016 ms | memory 🚚 | **1** (0.2% busy) |
| Decode, 64 users | 16 GB | 1 TFLOPs | **8 ms** | 1 ms | memory 🚚 | 64 (13% busy) |
| Prefill, 100 tokens | 16 GB | 1.6 TFLOPs | **8 ms** | 1.6 ms | memory 🚚 | 100 (20% busy) |
| Prefill, 2,000 tokens | 16 GB | 32 TFLOPs | 8 ms | **32 ms** | maths 👨‍🍳 | **2,000** (100% busy) |

The H100 can do **500 FLOPs per byte**. Below 500, memory is the limit. Above 500, maths is the limit.

### How the model makes a word

```
word → token ID → 4,096 numbers → × weights through 32 layers
     → a score for every word (128K) → softmax → chances → pick one
```

It gives **chances**, not one exact answer. That's why the same question can get different replies.

👉 **Prefill = few reads, lots of maths (maths-bound, GPU busy). Decode = one read per word, little maths (memory-bound, GPU waits).** A short prompt (under ~500 tokens) still behaves like decode.

**Check:** decode with 300 users. Memory or maths? → 300 < 500, so **memory** (reading 8 ms vs maths 4.8 ms)

## Step 4: Fixing decode (batching and smaller weights) 🔧

Decode is slow because **reading** is slow. Bus story: the trip (reading) takes 8 ms, and often only 1 person rides. So either **fill the bus** or **shorten the trip**.

### Fix 1: Batching 👥 (fill the bus)

> For each new word, the GPU has to load **all the model's weights** (16 GB) from memory. If 10 users are served **one by one**, the same weights are loaded **10 times** (160 GB), which wastes **time**, because reading memory is the slow part. With **batching**, the GPU loads the weights **once**, multiplies them with **all 10 users' inputs together**, and predicts the **next word for each user**. So for one step, there's **1 load instead of 10**, which is about **10× less reading**. The maths stays the same (each user still needs their own), but the maths is fast, so the whole step becomes much faster.

| Work (10 users, 1 word each) | One by one | Batched |
|---|---|---|
| Reading weights | 160 GB, 80 ms | **16 GB, 8 ms** ✅ |
| Maths | 10 × 16 GFLOPs | 10 × 16 GFLOPs (same) |

**How do different questions work in one go?** The weights are a **shared rule**, and each user's word is **their own input** (teacher story: "multiply by 3", where every student brings a different number). The GPU stacks the 10 users into a table, multiplies it by the same weights, and gets one output row per user. The rows never mix.

| Part | Shared? |
|---|---|
| Weights | ✅ read once, used by all |
| Current word (input) | ❌ each user's own |
| KV cache (chat memory) | ❌ each user's own, read separately |

- Batching saves **reading time**, not **storage space**. The weights are still 16 GB, and the KV cache grows with every user.
- One read = **1 word for each user**, not the whole answer.
- Users join at any time, and the server mixes them in each step (**continuous batching**, vLLM).

### Fix 2: Smaller weights 📦 (shorten the trip)

BF16 is the **normal** size. FP8 and INT4 are smaller, at a small accuracy cost.

| Weights | Size | Reading time (÷ 2,000 GB/s) |
|---|---|---|
| BF16 (normal) | 16 GB | 8 ms |
| FP8 (half) | 8 GB | **4 ms** |
| INT4 (quarter) | 4 GB | **2 ms** |

👉 **Decode is slow because of reading. Batching = fewer reads per word. Smaller weights = faster reads. Use both.**

**Check:** 1 user, INT4 weights. Words per second? → 4 ÷ 2,000 = 2 ms per word → **500 words/s**

## Step 5: Fixing prefill, and the KV cache ⚠️

### Why each job is slow

> **Prefill** is slow because of **too much maths**: many tokens share **one** read, but each token needs its own maths.
>
> **Decode** is slow because of **reading**: every word reads all the weights again, and the maths is tiny.

```
Prefill (2,000 tokens):  reading 8 ms  |  maths 32 ms     ← maths is slow
Decode  (1 word):        reading 8 ms  |  maths 0.016 ms  ← reading is slow
```

### Fixing prefill: faster maths 👨‍🍳

| Fix | What it does | 2,000-token prompt |
|---|---|---|
| **FP8 maths** | the H100 does FP8 maths **2× faster** (~2,000 TFLOPS) | 32 ms → **16 ms** |
| **More GPUs** | split the maths across GPUs | 2 GPUs → ~16 ms |
| **Prefix caching** | the same prompt start (like a system prompt) is computed **once** and reused | less maths |

This raises the **flat roof** and makes the first word appear sooner (**TTFT**). It doesn't help decode, because the maths was never decode's problem.

### The KV cache: what batching can't fix

- **KV cache** = the saved memory of the chat so far, a small note for every past token. **Each user has their own.**
- Llama 3 8B: **~0.13 MB per token** → 1K tokens = 0.13 GB, 8K = 1 GB, 32K = 4 GB per user.
- For **every new word**, the GPU reads:
  1. **Weights** (16 GB): **shared**, read once for all users ✅
  2. **Each user's KV cache**: **not shared**, read separately for each user ❌
- **Story:** the weights are the shared **textbook**. The KV cache is each student's **own notebook**. Batching saves the textbook reading, but every notebook must still be read.

```
64 users × 8K-token chats, one decode step:
Weights:  16 GB (read once, shared)
KV cache: 64 × 1 GB = 64 GB (each read separately)
Total:    80 GB  → the KV cache is 4× the weights
```

More users = **more** KV cache to read. The KV cache reads stay at ~1 FLOP per byte, so they're always on the slope.

| Fix (thinner notebooks) | Effect |
|---|---|
| **FP8 KV cache** | 2× smaller |
| **GQA** (already in Llama 3) | 4× smaller |
| **Shorter context** | fewer pages |

### The whole roofline in one rule 🏠

| Slow part | Where | Fix |
|---|---|---|
| 🚚 **Reading** (memory) | decode, short prompts, long KV cache | batching, smaller weights, smaller KV cache |
| 👨‍🍳 **Maths** | long prompts (prefill) | faster maths (FP8), more GPUs, prefix caching |

👉 **First find the slow part (reading or maths), then fix that part.**

**Check:** very long chats (32K tokens), and decode is slow even with batching. Why? → each user's KV cache (**4 GB**) is read for every word and isn't shared. Fix: FP8 KV cache or shorter context.

## Remember

| Step | Remember |
|---|---|
| 1 | **Intensity** = FLOPs per byte ≈ tokens sharing one read of the weights |
| 2 | Reading = 16 GB ÷ 2,000 GB/s = **8 ms**. Maths = tokens × 0.016 ms. The **bigger wins**. Switch point = **500** |
| 3 | **Prefill** = many tokens per read → maths-bound. **Decode** = 1 word per user per read → memory-bound |
| 4 | Fix decode: **batching** (fewer reads) + **smaller weights** (faster reads) |
| 5 | Fix prefill: **faster maths**. The **KV cache** is per user, so batching can't help it: shrink it instead |

## Quick check

1. Decode with 200 users, BF16 weights. Reading time? Maths time? Which is slow?
2. A 4,000-token prompt. Which is slow, reading or maths?
3. 10 users batched. How many times are the weights read for one step?
4. Does batching reduce the memory space the weights take?
5. Why does a long chat make decode slow, even with batching?

<details>
<summary>Answers</summary>

1. Reading 8 ms, maths 200 × 0.016 = 3.2 ms → **reading** (memory)
2. Reading 8 ms, maths 64 ms → **maths**
3. **Once** (1 word for each of the 10 users)
4. **No.** It saves **reading time**, not space.
5. Each user's **KV cache** is read for every word and **isn't shared**.

</details>
