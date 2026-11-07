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

## Step 2: What is throughput? 🏭

**Throughput = the total work the whole system finishes per second, for all users together** (tokens per second for the whole GPU).

**Story 🍕:** latency = how long **you** wait for your pizza. Throughput = how many pizzas the shop makes **per hour** for everyone. The customer cares about latency, and the owner cares about throughput 💰.

### How to calculate it

- **One step** = one read of the weights = **8 ms**.
- **Words per step** = how many new words come out of one read = **number of users** (1 word for each user).
- **Steps per second** = 1,000 ms ÷ 8 ms = **125** (always, because the weights are read once and shared).

```
Throughput = words per step × steps per second
```

| Users together | Words per step | Steps per second | **Throughput** |
|---|---|---|---|
| 1 | 1 | 125 | **125** words/s |
| 10 | 10 | 125 | **1,250** words/s |
| 64 | 64 | 125 | **8,000** words/s |

**3 users = 1 step, not 3 steps.** Users don't take turns; they share the same read:

```
Step 1 (8 ms) → user 1: "Hello", user 2: "Sure", user 3: "The"
Step 2 (8 ms) → word 2 for all 3 users
```

Steps are counted by **words written**, not by users.

**Story 🚌:** the bus leaves 125 times per second. 1 person per bus = 125 people/s. 64 per bus = 8,000 people/s. Same trips, just **fuller**.

| | Measures | Example |
|---|---|---|
| **Latency** | how long **one user** waits | 8 ms per word |
| **Throughput** | total words per second for **everyone** | 64 × 125 = 8,000 words/s |

The GPU costs the same per hour either way, so **64 users = 64× cheaper per word**.

👉 **Per user: still 125 words/s. Total: users × 125.**

**Check:** 100 users, 8 ms steps. Throughput? → 100 × 125 = **12,500 words/s**
