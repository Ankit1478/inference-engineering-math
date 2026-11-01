# KV Cache 5: Advanced

**When GPU memory runs out, what do you do with old K,V notes? This lesson covers offloading (move them to CPU memory). Eviction and disaggregation come next.**

## Quick summary (read this first)

1. GPU memory is **small** (80 GB on an H100). With many long chats, the KV cache **fills it up**.
2. The server also has **CPU memory (RAM)**, which is **much bigger** (500 GB–2 TB) but **slower**.
3. **Offloading** = move K,V notes that **aren't being used right now** from the GPU to the CPU.
4. When the user comes back, **copy them back** to the GPU instead of making K,V again.
5. It only helps when **copying back is faster than making K,V again**. For long chats it usually is.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Offloading deletes the K,V" | It **moves** them to CPU memory. They come back later. |
| "CPU memory is as fast as GPU memory" | It's **slower**. That's why only **unused** notes go there. |
| "Offloading is always faster" | Copying takes time too. It helps when copying back beats **making K,V again**. |

## Why this matters for inference

- When the GPU is full, new users must **wait**, or old notes must be **thrown away** and made again later (slow prefill).
- Offloading frees GPU space **without losing** the work already done.
- It helps most for **long chats** and **users who come back** after a pause.

## Real-life example

### Your desk and your cupboard 🗄️

- **Desk** = GPU memory: small, but everything is right in front of you.
- **Cupboard** = CPU memory: big, but you have to walk to it.

You keep today's work on the desk and put old files in the cupboard. When you need an old file, you **fetch** it. That's slower than the desk, but much faster than **writing the file again from scratch**.

## The math

### GPU vs CPU memory

| | GPU memory | CPU memory |
|---|---|---|
| Size | 80 GB | **500 GB–2 TB** 📦 |
| Speed | very fast ⚡ | slower 🐢 |

### A user goes quiet, then comes back

```
User goes away    → move their K,V: GPU ➡️ CPU   (frees GPU space for others)
User comes back   → move their K,V: CPU ➡️ GPU   (no need to make it again ✅)
```

| | Throw away | Offload |
|---|---|---|
| GPU space freed | ✅ | ✅ |
| When the user comes back | **make K,V again** (slow prefill) 😩 | **copy it back** (faster) ⚡ |

```
Offloading helps when:  time to copy back  <  time to make K,V again
```

## Inference connection

- vLLM and other servers can offload KV cache to CPU memory, and some can go further to disk.
- It works well with **prefix caching**: a popular system prompt's K,V can sit in CPU memory and come back when needed.

## Remember

1. GPU memory = **small and fast**. CPU memory = **big and slower**.
2. **Offloading** = move unused K,V to the CPU, and bring it back when needed.
3. Better than throwing away, because copying back is usually **faster than making K,V again**.

## Quick check

1. GPU memory is full, and a user has gone quiet for a while. What's better: throw away their K,V, or move it to CPU memory? Why?
2. Does offloading delete the K,V?

<details>
<summary>Answers</summary>

1. **Move it to CPU memory.** It frees the GPU, and when the user comes back, copying it back is faster than making K,V again.
2. **No.** It moves them to CPU memory so they can come back later.

</details>
