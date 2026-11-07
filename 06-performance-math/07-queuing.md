# 6.7 Basic queuing math

**Queuing answers one question: how many GPUs do I need so users don't wait too long? Story: a chai stall 🍵. Requests are customers, the GPU is the chai-wala. Example: Llama 3 8B on one H100, 8 ms per decode step, 200-word answers.**

| Step | Idea |
|---|---|
| 1 | **Arrival rate vs service rate**: requests coming in vs requests the GPU finishes |
| 2 | **Busy (utilization)**: arrivals ÷ capacity. At 100% or more, the line grows forever |
| 3 | **Why the wait explodes near 100%**: busy ÷ (1 − busy) |
| 4 | **Little's Law** ⭐: users inside = arrival rate × time stayed |
| 5 | **How many GPUs**: plan for about 75% busy, not 100% |

## Step 1: Arrival rate vs service rate 🚶

There are two numbers:

- **Arrival rate**: how many requests come in each second (customers walking in)
- **Service rate**: how many requests the GPU can finish each second (how fast the chai-wala works)

**Working out the service rate:**

- Each user stays on the GPU for 200 words × 8 ms = **1.6 s**
- Say the SLO lets us seat **64 users** in the batch
- The GPU finishes 64 users every 1.6 s → 64 ÷ 1.6 = **40 requests per second**

```
Service rate = seats ÷ time each user stays = 64 ÷ 1.6 = 40 req/s
```

## Step 2: How busy is the GPU? (utilization) 📊

```
Busy = arrival rate ÷ service rate
```

| Requests coming in | Busy | What happens |
|---|---|---|
| 20 per second | 50% | Relaxed, almost no line |
| 36 per second | 90% | Works, but there's a line |
| 40 per second | 100% | Just barely keeping up |
| 50 per second | 125% 💥 | 10 extra requests pile up **every second**, so the line grows forever |

**Rule:** busy must stay below 100%, or users wait forever.

## Step 3: Why the wait explodes near 100% 📈

Arrivals are random, not steady. Sometimes 5 people walk in at once. If the GPU has spare room, it catches up. If it's always busy, it never catches up.

```
Wait in line ≈ busy ÷ (1 − busy) × 25 ms
```

The 25 ms is how often the GPU frees up a seat: 1 ÷ 40 per second.

| Busy | Wait in line |
|---|---|
| 50% | 1 × 25 = **25 ms** 😊 |
| 80% | 4 × 25 = **100 ms** |
| 90% | 9 × 25 = **225 ms** 😐 |
| 99% | 99 × 25 = **~2.5 s** 😱 |

Going from 90% to 99% busy only adds 10% more work, but the wait gets **11× longer**.
This waiting time **adds directly to TTFT**, so the user feels it before the first word.

> This is a rough model. Real servers aren't exactly like this, but the shape (flat, then a sudden explosion near 100%) is the same.

## Step 4: Little's Law ⭐

```
Users inside = arrival rate × time each one stays
```

At the chai stall: if 2 customers arrive every minute and each stays 5 minutes, there are about 10 people inside at any time.

**On the GPU:** 30 requests per second × 1.6 s each = **48 users** in the batch at once.

**Why it matters:** it tells you the **KV-cache memory** you need.

- 1 user = 700 tokens (500 prompt + 200 answer) × 0.13 MB ≈ **0.09 GB**
- 48 users × 0.09 GB ≈ **4.3 GB of KV cache**
- An H100 has about 56 GB free for KV, so it fits easily ✅

(Step 1 was Little's Law the other way round: 64 seats ÷ 1.6 s = 40 req/s.)

## Step 5: How many GPUs do I need? 🖥️

**Target:** 50 requests per second.

- **1 GPU:** 50 ÷ 40 = 125% busy 💥 The line grows forever.
- **Don't plan for 100%.** Plan for about **75% busy** so random bursts don't explode the wait.
  - At 75%, each GPU handles 40 × 0.75 = **30 req/s**

```
GPUs = arrival rate ÷ (one GPU's capacity × target busy), rounded up
     = 50 ÷ (40 × 0.75) = 1.67 → 2 GPUs
```

**Check with 2 GPUs:**

- Each GPU gets 25 req/s, so it's 62.5% busy
- Wait in line = 0.625 ÷ 0.375 × 25 ≈ **42 ms** ✅ Inside the SLO.

**Why real companies run at about 70–80%, not 100%:** the spare 20–30% soaks up traffic spikes. Running at 100% looks efficient on paper, but users would wait seconds.

## Remember

1. **Busy = arrivals ÷ capacity.** At 100% or more, the line grows forever.
2. **Wait ≈ busy ÷ (1 − busy).** It's small until about 80% busy, then it explodes.
3. **Little's Law: users inside = arrival rate × time stayed.** Use it to size the batch and the KV-cache memory.
4. **GPUs = arrivals ÷ (capacity × ~0.75)**, rounded up.
5. Queue wait adds to **TTFT**.

## Quick check

1. A GPU can finish 40 req/s and 44 req/s arrive. What happens?
2. 20 req/s arrive and each stays 2 s. How many users are inside?
3. Going from 80% to 90% busy, how much longer is the wait?
4. 100 req/s arrive, one GPU does 40 req/s, and you target 75% busy. How many GPUs?

<details>
<summary>Answers</summary>

1. **110% busy**, so the line grows by **4 every second, forever**.
2. 20 × 2 = **40 users**.
3. From 4× to 9×, so about **2.25× longer**.
4. 100 ÷ 30 = 3.3, so **4 GPUs**.

</details>
