# 6.5 Arithmetic intensity and the roofline model ⭐

**The roofline answers one question: is the GPU slow because of memory (reading weights) or because of maths? It all comes from `time = distance ÷ speed`. Example: Llama 3 8B on one H100.**

| Step | Idea |
|---|---|
| 1 | **Answers per read** (arithmetic intensity) |
| 2 | **The speed limit**: fill the bus (500 seats) |

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
