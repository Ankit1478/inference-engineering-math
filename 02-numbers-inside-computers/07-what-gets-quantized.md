# 2.7 What gets quantized: weights, activations, KV cache

**A model has 3 big groups of numbers. We can make each of them smaller, but each one is a different amount of work and gives a different win.**

## Quick summary (read this first)

1. There are **3 things** we can quantize: **weights**, **activations** and the **KV cache**.
2. **Weights** = the model's fixed numbers. They're **the easiest** to quantize and save the most memory on a big model.
3. **Activations** = temporary numbers made while the model runs. They're **the hardest** because they change with every input and have giant outliers.
4. **KV cache** = the saved K,V notes. Quantizing it means **more users and longer chats**.
5. Names: **W8A8** = 8-bit weights and 8-bit activations. **W4A16** = 4-bit weights and 16-bit activations.
6. **Rule:** quantize whatever takes up the most memory.

### The 3 things

| What | What it is | Changes? | How hard to quantize |
|---|---|---|---|
| **Weights** | The model's learned numbers | ❌ Fixed after training | 🟢 Easy |
| **Activations** | Numbers passing between layers while it runs | ✅ New for every input | 🔴 Hard |
| **KV cache** | Saved K,V of every word so far | ✅ Grows with words × users | 🟡 Medium |

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Quantized model" means everything is 8-bit | Usually only **some parts** are. Read the name: W4A16 means the activations are still 16-bit. |
| Weights are always the biggest part | For a **small model with many users**, the **KV cache** can be bigger. |
| W4A16 makes the maths faster | It doesn't. The maths still runs in BF16. It's faster only because there are **fewer bytes to read**. |
| Activations are easy because they're temporary | They're the **hardest**: they change every time and have giant outliers. |

## Why this matters for inference

- When someone says "let's quantize the model", the first question is **which part?**
- Each part fixes a different problem:
  - **Weights** → the model doesn't fit, or decode is slow.
  - **Activations** → the maths (prefill, big batches) is slow.
  - **KV cache** → there's no room for more users or longer chats.

## Real-life example

### Moving house 📦

| Thing | Like | How easy to pack small |
|---|---|---|
| **Weights** | Your furniture | Easy. It doesn't move, so you can measure it once and pack it well. |
| **Activations** | Things people keep carrying in and out | Hard. Every box is different, and sometimes someone brings a sofa. |
| **KV cache** | The pile of letters that grows every day | Medium. It keeps growing, so it matters more the longer you stay. |

## The math

### 1. Weights: easy

Weights are fixed, so we quantize them **once, before serving**. We have time to pick good scales and test the quality.

**W4A16** (AWQ, GPTQ):

```
stored:  weights in INT4   (½ byte each)
maths:   turn them back into BF16 → multiply in BF16
```

The maths isn't faster, but decode is limited by **reading memory**, and there are 4× fewer bytes to read. So decode gets faster.

### 2. Activations: hard

Activations change with every input, and a few of them are **giant outliers**. Here's what happens with INT8 (from 2.4 and 2.5):

```
activations:  0.5   -1.2   0.8   60      ← one giant number
scale:        60 ÷ 127 = 0.47

0.5  → round(1.06)  = 1  → back = 0.47   (was 0.5)
-1.2 → round(-2.54) = -3 → back = -1.42  (was -1.2)  ❌ 18% off
0.8  → round(1.69)  = 2  → back = 0.94   (was 0.8)   ❌ 18% off
```

One giant number makes the steps big, so all the small numbers get rough.

How people fix it:
- **FP8** handles this better than INT8, because it keeps small decimals (2.6).
- **SmoothQuant** moves the difficulty from the activations into the weights, because weights are easier to quantize.

**Why bother?** If **both** weights and activations are 8-bit (**W8A8**), the GPU does the maths itself in 8-bit, which is **about 2× faster** on H100 Tensor Cores. That's the only way to speed up the **maths** part (prefill and big batches).

### 3. KV cache: medium

From the KV cache deep dive: `--kv-cache-dtype fp8` stores K,V in 1 byte instead of 2, so the cache is **2× smaller** → 2× more users or 2× longer chats.

### What usually stays in BF16

Small but sensitive parts: **RMSNorm, softmax**, and often the **first layer (embeddings)** and **last layer (lm_head)**. They're tiny, so keeping them in BF16 costs almost nothing.

### Which one helps what

| Quantize | Less memory | Faster decode (memory-bound) | Faster prefill (compute-bound) |
|---|---|---|---|
| Weights only (**W4A16**) | ✅ big | ✅ | ❌ |
| Weights + activations (**W8A8**) | ✅ | ✅ | ✅ |
| **KV cache** (FP8) | ✅ more users | ✅ for long chats | ❌ |

## Inference connection

### Worked example: Llama 3 8B on one H100 (80 GB), 8K words per user

From earlier: weights = **16 GB** in BF16, KV cache = **1 GB per user**, and we keep about **8 GB** for extra.

```
users = (80 − weights − 8) ÷ KV per user
```

| Setup | Weights | Room for KV | KV per user | Users |
|---|---|---|---|---|
| All BF16 | 16 GB | 56 GB | 1 GB | **56** |
| FP8 weights only | 8 GB | 64 GB | 1 GB | **64** (+14%) |
| FP8 KV cache only | 16 GB | 56 GB | 0.5 GB | **112** (+100%) 🚀 |
| Both | 8 GB | 64 GB | 0.5 GB | **128** |

👉 For a **small model with many users**, the KV cache is the big part, so quantizing **the KV cache** helps more.
👉 For a **big model** (70B = 140 GB in BF16), the **weights** are the big part, so quantize those first.

### In vLLM

```
--quantization fp8          # W8A8 in FP8: weights + activations
--quantization awq          # load an AWQ model: W4A16
--kv-cache-dtype fp8        # KV cache in FP8
```

## Remember

- 3 things: **weights** (easy), **activations** (hard), **KV cache** (medium).
- **W4A16** = less memory and faster decode, but the maths isn't faster.
- **W8A8** = the maths itself gets faster too (prefill and big batches).
- Activations are hard because of **giant outliers**. FP8 and SmoothQuant help.
- **Quantize the biggest part first**: weights for big models, the KV cache for many users.
- Always **test quality** (2.6).

## Quick check

1. Which of the 3 is easiest to quantize, and why?
2. What does **W4A16** mean?
3. Your prefill is too slow (compute-bound). Will W4A16 help?
4. Llama 3 8B with 100 users on one GPU, and memory is full. Weights or KV cache first?
5. Why do giant outliers make activations hard to quantize?

<details>
<summary>Answers</summary>

1. **Weights.** They're fixed, so you quantize them once, before serving, and can test the quality.
2. **4-bit weights, 16-bit activations.**
3. **No.** The maths still runs in BF16. You need **W8A8** to make the maths faster.
4. **KV cache.** 100 users × 1 GB = 100 GB of KV vs only 16 GB of weights.
5. One giant number sets the scale, so the steps get big and all the small numbers become rough.

</details>
