# 6.3 Counting memory bytes ⭐

**Memory decides whether a model fits, how many users fit, and how fast each token is. This lesson goes step by step on one real example: Llama 3 8B (then 70B).**

| Step | Idea |
|---|---|
| 1 | Memory for the **weights** |
| 2 | Memory for the **KV cache** (one user) |
| 3 | How many **users** fit on one GPU |
| 4 | **Bytes read** for each token (speed) |
| 5 | A big model: how many **GPUs**? |

## Step 1: Memory for the weights

Every weight is a number, and every number takes some **bytes** (Section 2).

```
Weights memory = number of weights × bytes per weight
```

**Llama 3 8B:**

| Format | Bytes per weight | Memory |
|---|---|---|
| FP32 | 4 | 8B × 4 = **32 GB** |
| **BF16** (normal) | 2 | 8B × 2 = **16 GB** |
| FP8 | 1 | 8B × 1 = **8 GB** |
| INT4 | 0.5 | 8B × 0.5 = **4 GB** |

👉 Same model, different size. **We choose the bytes.**

| GPU | Memory | Llama 3 8B in BF16 (16 GB) |
|---|---|---|
| RTX 4090 | 24 GB | ✅ fits |
| Laptop GPU | 8 GB | ❌ needs INT4 (4 GB) |
| H100 | 80 GB | ✅ fits, lots of room left |

**Check:** a 13B model in BF16 on a 24 GB GPU? → 26 GB, **doesn't fit** (FP8 = 13 GB fits)

## Step 2: Memory for the KV cache (one user)

While writing, the model **saves K and V for every token**. These saved notes take memory too.

### A. Memory for 1 token

```
KV per token = 2 × layers × KV heads × head size × bytes
```

| Piece | Llama 3 8B | Why |
|---|---|---|
| 2 | 2 | one K **and** one V |
| layers | 32 | each layer saves its own |
| KV heads | 8 | GQA (5.3) |
| head size | 128 | numbers per head |
| bytes | 2 | BF16 |

```
2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB per token
```

### B. Memory for 1 user

```
KV per user = KV per token × tokens in the chat
```

| Chat length | KV cache |
|---|---|
| 1,000 tokens | ≈ **128 MB** |
| 8,000 tokens | ≈ **1 GB** |
| 128,000 tokens | ≈ **16 GB** 😱 as big as the whole model |

**Check:** a 4,000-token chat? → 128 KB × 4,000 ≈ **0.5 GB**

## Step 3: How many users fit on one GPU?

GPU memory is a **suitcase** 🧳:

```
GPU memory = weights + KV cache (all users) + extra (~10% working space)
```

**Llama 3 8B on an H100 (80 GB), each user at 8,000 tokens:**

```
Suitcase:          80 GB
− Weights (BF16):  16 GB
− Extra (~10%):     8 GB
= Left for KV:     56 GB

Users = 56 GB ÷ 1 GB = 56 users
```

| Chat length | KV per user | Users that fit |
|---|---|---|
| 2,000 tokens | 0.25 GB | **224** |
| 8,000 tokens | 1 GB | **56** |
| 32,000 tokens | 4 GB | **14** |

👉 Longer chats = fewer users per GPU = more expensive per user.

| Trick | Effect at 8K tokens |
|---|---|
| FP8 KV cache (0.5 GB per user) | 56 → **112 users** |
| FP8 weights (room for KV → 64 GB) | 56 → **64 users** |

**Check:** every user at 16,000 tokens? → 2 GB each, 56 ÷ 2 = **28 users**

## Step 4: Bytes read for each token (speed)

In every decode step, the GPU reads:

1. **The weights**: once, **shared** by all users in the batch.
2. **Each user's KV cache**: **separate** for every user.

```
Bytes read per step = weights + (KV per user × users)
Time per step       = bytes read ÷ memory speed (2,000 GB/s)
```

📖 **Story:** the textbook (weights) is shared, but each student also has **their own notebook** (KV cache) that must be read too.

**Llama 3 8B, each user at 8,000 tokens (1 GB KV):**

```
1 user:    16 + 1      = 17 GB → 17 ÷ 2,000 = 8.5 ms per step
56 users:  16 + 56 × 1 = 72 GB → 72 ÷ 2,000 = 36 ms per step
```

| | 1 user | 56 users |
|---|---|---|
| Time per step | 8.5 ms | 36 ms |
| Speed for **each** user | ~118 tokens/sec | ~28 tokens/sec |
| Speed for the **whole GPU** | ~118 tokens/sec | **~1,550 tokens/sec** 🚀 |

👉 Batching makes the GPU **13× more productive**, but each user gets a bit slower.

**Surprise:** with 56 users, the reads are 16 GB of weights (22%) and **56 GB of KV caches (78%)**. At long chats, reading the KV cache is the main cost, which is why FP8 KV and GQA matter so much.

**Check:** 10 users at 8,000 tokens? → 16 + 10 = 26 GB → **13 ms** per step

