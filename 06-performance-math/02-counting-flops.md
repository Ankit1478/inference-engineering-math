# 6.2 Counting FLOPs ⭐

**A FLOP is one small maths action. A model needs about 2 FLOPs per weight per token. This lesson goes step by step on one real example: Llama 3 8B.**

| Step | Idea |
|---|---|
| 1 | 1 weight = **2 FLOPs** |
| 2 | FLOPs = **2 × weights × tokens** |
| 3 | Time = FLOPs ÷ GPU speed, and why real decode is slower (memory) |
| 4 | Prefill sets **TTFT** |
| 5 | **MFU**: how busy the GPU really is |

## Step 1: What is a FLOP?

A **FLOP** is **one small maths action**: one multiply **or** one add.

```
3 × 4 = 12     → 1 FLOP (multiply)
12 + 5 = 17    → 1 FLOP (add)
```

So `3 × 4 + 5` = **2 FLOPs**.

### The pattern inside a model

A model does this again and again: **take a number, multiply it by a weight, add it to a total.**

```
total = total + (input × weight)   ← 1 multiply + 1 add = 2 FLOPs
```

So **every weight costs 2 FLOPs** each time it's used.

| Weights used | FLOPs |
|---|---|
| 1 | 2 |
| 10 | 20 |
| 1,000 | 2,000 |

**Check:** a tiny layer has 500 weights. FLOPs to use them all once? → **1,000**

## Step 2: FLOPs to write one word

To write **one token**, the model uses **every weight once**. Llama 3 8B has **8 billion weights**.

```
FLOPs for 1 token = 2 × 8 billion = 16 billion FLOPs = 16 GFLOPs
```

| Number | Name |
|---|---|
| 1,000,000,000 (billion) | **G** (giga) |
| 1,000,000,000,000 (trillion) | **T** (tera) |

| Model | Weights | FLOPs per token |
|---|---|---|
| Llama 3 8B | 8B | **16 GFLOPs** |
| 13B model | 13B | 26 GFLOPs |
| Llama 3 70B | 70B | 140 GFLOPs |

### Many tokens

```
FLOPs = 2 × weights × tokens
```

10 tokens on Llama 3 8B, two ways, same answer:

```
16 GFLOPs × 10 = 160 GFLOPs
2 × 8B × 10 = 20 × 8B = 160 GFLOPs ✅
```

(Notes: MoE models count only the **active** weights. Training costs 6 × weights per token, but inference is 2 ×.)

**Check:** 100 tokens on Llama 3 8B? → 2 × 8B × 100 = **1.6 TFLOPs**

## Step 3: How long does it take on a GPU?

Like a car trip, `time = distance ÷ speed`:

```
Time = FLOPs ÷ GPU speed (FLOPS)
```

**FLOPs** = the amount of maths. **FLOPS** (capital S) = maths **per second**. An H100 does about **1,000 TFLOPS** (rounded).

Llama 3 8B, 1 token:

```
0.016 TFLOPs ÷ 1,000 TFLOPS = 0.000016 s = 0.016 ms ⚡
```

That would mean **~62,000 tokens per second**. Real speed for one user is only **~125**. Why?

### The missing part: reading the weights 📖

**Story:** a student must **re-read the whole textbook** before answering each question.

| Job | Time |
|---|---|
| Re-read the book | 8 minutes |
| Think and write | 1 second |

One answer ≈ **8 minutes**. Fast thinking doesn't help, because **reading** takes all the time.

**The GPU is the same:**

| Story | GPU (Llama 3 8B) |
|---|---|
| Textbook | weights = 8B × 2 bytes = **16 GB** |
| Re-read the book | read 16 GB from memory at ~2,000 GB/s = **8 ms** |
| Think and write | the maths = **0.016 ms** |

```
1 token ≈ 8 ms  →  1,000 ÷ 8 ≈ 125 tokens per second
```

Reading takes **500× longer** than the maths. This is called **memory-bound** (more in 6.4).

**The fix: batching 👥.** 10 students ask at once, so you read the book **once** and answer all 10. The server reads the weights once and serves many users.

**Check:** a 70B model in BF16 = 140 GB. One decode token at 2,000 GB/s? → 140 ÷ 2,000 = **70 ms** (~14 tokens/sec)

