# KV Cache 5: Advanced

**When GPU memory runs out, or big jobs get in the way, what do you do? This lesson covers offloading (move K,V to CPU memory), eviction (choose what to throw away) and disaggregation (prefill and decode on different GPUs).**

## Quick summary (read this first)

**Offloading**

1. GPU memory is **small** (80 GB on an H100). The server's **CPU memory** is **much bigger** (500 GB–2 TB) but **slower**.
2. **Offloading** = move K,V that **isn't being used right now** from the GPU to the CPU, and **copy it back** when needed.
3. It helps when **copying back is faster than making K,V again**. For long chats it usually is.

**Eviction**

4. **Eviction** = choosing **which K,V to throw away** from the KV cache when memory is full.
5. Usual rule: **LRU (Least Recently Used)**. Throw away the notes **used longest ago**.
6. If even active users don't fit, the server **pauses** one user (usually the newest) and **evicts** or **offloads** their K,V. It continues later.
7. Only the **K,V numbers** are lost. The **chat text** is still in your app, so the K,V can always be **made again**.

**Disaggregation**

8. Prefill (heavy maths) and decode (memory reading) **get in each other's way** on the same GPU. A big prefill makes everyone's answer **stutter**.
9. **Disaggregation** = **prefill GPUs** make the K,V, then **send the KV cache** to **decode GPUs**, which write the answer.
10. Smooth answers, but the KV cache must travel between GPUs, so it needs a **fast connection**.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Offloading deletes the K,V" | It **moves** them to CPU memory. They come back later. |
| "CPU memory is as fast as GPU memory" | It's **slower**. That's why only **unused** notes go there. |
| "Offloading is always faster" | Copying takes time too. It helps when copying back beats **making K,V again**. |
| "Eviction deletes the user's chat" | It deletes only the **K,V numbers** on the GPU. The **chat text** is safe in the app. |
| "The KV cache stores the words (text)" | It stores **K,V numbers made from** the words. The text is kept by the **app**. |
| "The KV cache stores only the question" | It stores K,V for **every word seen so far**: system prompt, history, question **and the answer words already written**. |
| "Eviction throws away the newest notes" | **LRU** throws away the notes **used longest ago**. |
| "Disaggregation removes the KV cache" | The KV cache is still made, then **sent** from the prefill GPU to the decode GPU. |

## Why this matters for inference

- When the GPU is full, new users must **wait**. Offloading and eviction make space.
- Smart eviction keeps the **useful** notes (like a popular system prompt) and drops the old ones.
- Disaggregation keeps answers **smooth** even when someone pastes a huge document. Big AI companies use it at scale.

## Real-life example

### Offloading: your desk and your cupboard 🗄️

- **Desk** = GPU memory: small, but everything is right in front of you.
- **Cupboard** = CPU memory: big, but you have to walk to it.

You put old files in the cupboard and **fetch** them when needed. Slower than the desk, but much faster than **writing the file again from scratch**.

### Eviction: your phone's photo gallery 📱

The phone is full. You delete photos you **haven't opened in years**, not the one you took today.

### Text vs KV cache: the book and your notes 📚

- **Text** = the actual book, kept safely in the library (your app).
- **KV cache** = your **notes about the book** (server's GPU memory).

If you lose the notes, you read the book again and rewrite them. Slower, but nothing is lost.

### Disaggregation: a restaurant kitchen 🍳

- **Chopping station** (prefill): cuts all the vegetables at once. Heavy work.
- **Cooking station** (decode): cooks the dish step by step.

If one cook does both, the cooking **stops** every time a big chopping job comes in. Two stations means **nobody waits**.

## The math

### 1. Offloading

| | GPU memory | CPU memory |
|---|---|---|
| Size | 80 GB | **500 GB–2 TB** 📦 |
| Speed | very fast ⚡ | slower 🐢 |

```
User goes away    → move their K,V: GPU ➡️ CPU   (frees GPU space for others)
User comes back   → move their K,V: CPU ➡️ GPU   (no need to make it again ✅)
```

```
Offloading helps when:  time to copy back  <  time to make K,V again
```

### 2. What's in the KV cache for one chat

```
You ask: "What is AI?"

Prefill:  K,V for  What, is, AI, ?     → writes "AI"
Decode 1: + K,V for "AI"               → writes "is"
Decode 2: + K,V for "is"               → writes "a"

KV cache now: [What][is][AI][?][AI][is]
               └── question ──┘└answer┘
```

| Part | In the KV cache? |
|---|---|
| System prompt, chat history, new question | ✅ (as K,V numbers) |
| Answer words written so far | ✅ |
| The next word (not written yet) | ❌ that's what it's predicting |

```
Words in the KV cache = words read + words written so far
```

### 3. Eviction (LRU)

```
Saved notes:          last used
"System prompt A"     1 second ago   ✅ keep
"Riya's old chat"     10 min ago     ✅ keep
"Aman's old chat"     2 hours ago    🗑️ throw away first
```

```
Before:  [Sys prompt][Riya][Aman's old chat]  ← full ❌
After:   [Sys prompt][Riya][  free page  ]    ← new user fits ✅
```

| What gets thrown away | When |
|---|---|
| Old notes nobody is using (LRU) | memory full, normal case ✅ |
| A paused user's notes | even active users don't fit |
| Unimportant words inside one long chat | research tricks, can hurt quality ⚠️ |

**What happens when Aman comes back?**

```
Evicted   → app sends the chat text again → make K,V again (slow prefill) 😩
Offloaded → copy K,V back from CPU memory (faster) ⚡
```

| | Removed from the GPU? | Comes back by |
|---|---|---|
| **Eviction** | ✅ deleted | **making it again** from the text |
| **Offloading** | ✅ moved to CPU | **copying it back** |

### 4. Disaggregation

**Same GPU:**

```
GPU: [decode][decode][ PREFILL 10,000 words ████████ ][decode]
                              ↑
            all users' answers freeze for a moment 🥶
```

**Split:**

```
New question → 🟦 Prefill GPU: reads the question, makes the K,V
                       ↓ sends the KV cache over 📦
               🟩 Decode GPU: writes the answer, word by word
```

| | Same GPU | Disaggregated |
|---|---|---|
| Answers stutter? | yes, when a big prefill comes 🥶 | **no**, smooth ✅ |
| Tune each part | hard | **easy** (e.g. 2 prefill GPUs + 6 decode GPUs) |
| Extra cost | none | **send the KV cache** between GPUs 📦 |

## Inference connection

| Trick | Solves |
|---|---|
| Offloading | GPU full, but don't want to lose work |
| Eviction (LRU) | GPU full, must free space |
| Disaggregation | prefill and decode slowing each other down |

- vLLM uses LRU to evict old **prefix cache** pages, and pauses users when memory runs out.
- Offloading works well with **prefix caching**: a popular system prompt's K,V can sit in CPU memory and come back when needed.
- Disaggregation needs a **fast connection** (NVLink or a fast network) to send the KV cache.

## Remember

1. **Offloading** = move unused K,V to CPU memory and bring it back later.
2. **Eviction** = throw away K,V to make space. **LRU**: oldest-used goes first.
3. The KV cache stores **K,V numbers** for **every word seen so far** (question **and** answer so far). The **text** lives in the app.
4. Evicted K,V can always be **made again** from the text. It's just slower.
5. **Disaggregation** = prefill GPUs make the K,V, decode GPUs write the answer. Smooth answers, but the KV cache must be sent between them.

## Quick check

1. GPU memory is full, and a user has gone quiet for a while. What's better: throw away their K,V, or move it to CPU memory? Why?
2. Does offloading delete the K,V?
3. Saved notes were last used 5 seconds, 1 hour and 3 days ago. Which does LRU throw away first?
4. Aman's K,V was evicted, and he sends a new message. Is his chat lost?
5. The model has read a 5-word question and written 3 answer words. How many words' K,V are in the KV cache?
6. In disaggregation, which GPU makes the KV cache, and which uses it?

<details>
<summary>Answers</summary>

1. **Move it to CPU memory.** It frees the GPU, and copying it back is faster than making K,V again.
2. **No.** It moves them to CPU memory so they can come back later.
3. The one used **3 days ago**.
4. **No.** The app still has the text. The server just makes the K,V again (extra work).
5. 5 + 3 = **8**
6. The **prefill GPU** makes it. The **decode GPU** uses it to write the answer.

</details>
