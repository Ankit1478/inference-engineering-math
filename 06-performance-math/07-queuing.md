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
