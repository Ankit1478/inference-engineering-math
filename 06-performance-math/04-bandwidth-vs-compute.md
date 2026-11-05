# 6.4 Memory bandwidth vs compute speed

**A GPU has two speeds: how fast it reads memory and how fast it does maths. Every job waits for whichever is slower. Decode waits for memory, and prefill waits for maths.**

Example throughout: **Llama 3 8B on an H100**.

| Step | Idea |
|---|---|
| 1 | A GPU has **two speeds** |
| 2 | Two times: the **bigger** one wins |
| 3 | Decode vs prefill |
| 4 | The **switch point** (~500) |
| 5 | Which **fix** for which problem |

## Step 1: A GPU has two speeds

Remember the student story 📖: **reading** the book and **thinking**.

| Speed | Measures | Unit | H100 (round number) |
|---|---|---|---|
| 🚚 **Bandwidth** | reading memory | GB/s | **2,000 GB/s** |
| 👨‍🍳 **Compute** | doing maths | FLOPS | **1,000 TFLOPS** |

```
Reading:  2 trillion bytes per second
Maths:    1,000 trillion FLOPs per second

Maths ÷ reading = 1,000 ÷ 2 = 500
```

👉 The GPU can do **500 maths actions** in the time it reads **1 byte**. To keep the chef busy, every byte delivered must be used about 500 times.

(The real H100 is ~3,350 GB/s and ~990 TFLOPS, a ratio of ~300. We use round numbers.)

## Step 2: Two times. The bigger one wins

```
Memory time = bytes ÷ 2,000 GB/s
Maths time  = FLOPs ÷ 1,000 TFLOPS
Real time   ≈ the BIGGER of the two
```

🍕 **Pizza order:** if the dough takes 8 minutes to arrive and baking takes 1 second, you wait **8 minutes**. A faster oven doesn't help.

## Step 3: Decode vs prefill

| | Decode (1 user, 1 token) | Prefill (2,000-token prompt) |
|---|---|---|
| Bytes read | 16 GB (weights) | 16 GB (weights, read **once**) |
| Memory time | 16 ÷ 2,000 = **8 ms** | **8 ms** |
| FLOPs | 2 × 8B = 16 GFLOPs | 2 × 8B × 2,000 = 32 TFLOPs |
| Maths time | **0.016 ms** | **32 ms** |
| Bigger one | memory | maths |
| Name | **memory-bound** 🚚 | **compute-bound** 👨‍🍳 |

👉 **Decode waits for memory. Prefill waits for maths.**

## Step 4: The switch point

Batch decode adds users. The weights are read **once**, but the maths grows with every user:

```
Memory time:  8 ms (same)
Maths time:   users × 0.016 ms
```

| Users | Memory time | Maths time | Limit |
|---|---|---|---|
| 1 | 8 ms | 0.016 ms | memory |
| 64 | 8 ms | 1 ms | memory |
| **500** | 8 ms | **8 ms** | **equal** ⚖️ |
| 1,000 | 8 ms | 16 ms | maths |

👉 The switch is at **500**, the same number as Step 1. Every user is one more maths job for each byte read.

**In real life** each user also brings their **own KV cache** to read (6.3 Step 4), so memory time grows too. Decode usually stays **memory-bound**, especially with long chats.

The same rule works for prefill: a prompt **over ~500 tokens** is compute-bound. Under that, it's memory-bound.

## Step 5: Which fix for which problem

| Problem | Meaning | Fixes |
|---|---|---|
| 🚚 **Memory-bound** (decode) | too many bytes to read | Fewer bytes: **FP8/INT4 weights**, **FP8 KV**, GQA. More reuse: **batching** |
| 👨‍🍳 **Compute-bound** (prefill) | too much maths | Faster maths: **FP8 (W8A8)** on Tensor Cores. Less maths: **prefix caching**. More GPUs |

**Trap:** W4A16 (INT4 weights, BF16 maths) speeds up **decode** (fewer bytes) but **not prefill** (same maths). See 2.7.

## Remember

- 2 speeds: **bandwidth** (reading) and **compute** (maths). H100 ratio ≈ **500**.
- Real time ≈ the **bigger** of memory time and maths time.
- **Decode = memory-bound. Prefill = compute-bound.**
- The switch is at **~500 tokens or users** per weight read.
- Memory-bound → **fewer bytes**. Compute-bound → **less or faster maths**.

## Quick check

1. A GPU reads 1,000 GB/s and does 500 TFLOPS. Its ratio?
2. Llama 3 8B, prefill of a 100-token prompt: memory time, maths time, which limit?
3. Same model, prefill of 4,000 tokens: which limit?
4. Decode is slow. Will a GPU with **faster maths but the same memory speed** help?
5. Will W4A16 make prefill faster?

<details>
<summary>Answers</summary>

1. 500 trillion ÷ 1 trillion = **500**
2. Memory 8 ms, maths 1.6 ms → **memory-bound** (under 500 tokens)
3. Maths 64 ms vs memory 8 ms → **compute-bound**
4. **No.** Decode waits for memory, so a faster oven doesn't help.
5. **No.** The maths still runs in BF16.

</details>
