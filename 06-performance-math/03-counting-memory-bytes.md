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

