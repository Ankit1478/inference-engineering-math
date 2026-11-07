# 6.6 Latency vs throughput ⭐

**Latency = how long ONE user waits. Throughput = how much the whole system finishes per second, for ALL users. Bigger batches raise throughput but, after a point, slow every user down. Example: Llama 3 8B on one H100.**

| Step | Idea |
|---|---|
| 1 | **Latency**: one user's wait (TTFT + TPOT) |
| 2 | **Throughput**: total words per second for everyone |
| 3 | **Why they fight**: bigger batch = more total, slower each |
| 4 | **SLO and goodput**: the promise that sets the batch size |

## Step 1: What is latency? ⏱️

**Latency = how long ONE user waits.**

**Story 🍕:** you order a pizza. Latency = how long **you** wait until you get it. It's about one person, not how many pizzas the shop makes.

| Wait | Name | Meaning |
|---|---|---|
| ⏳ Until the **first word** appears | **TTFT** (Time To First Token) | the prefill time |
| ✍️ Between **each next word** | **TPOT** (Time Per Output Token) | the decode time per word |

```
You press Enter → [TTFT] → "Hello" → [TPOT] → "how" → [TPOT] → "are" → ...
```

Llama 3 8B, 1 user, H100: TTFT (2,000-token prompt) ≈ **32 ms**, TPOT (reading 16 GB) = **8 ms**.

```
Total latency = TTFT + (words × TPOT)
100-word reply = 32 + (100 × 8) = 832 ms ≈ 0.8 s
```

| Number | What the user feels |
|---|---|
| **TTFT** | "Did it start?" |
| **TPOT** | "How fast does it type?" (8 ms = 125 words/s, much faster than you read) |

👉 **Latency = TTFT + time for each word. Lower is better.**

**Check:** TTFT 50 ms, TPOT 10 ms, 200-word reply? → 50 + 2,000 = **2,050 ms ≈ 2 s**
