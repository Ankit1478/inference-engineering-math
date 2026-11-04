# 3.4 Sin and cos (RoPE)

**RoPE tells the model where each word is by rotating its Q and K vectors. The further along the word is, the more it's turned.**

## Quick summary (read this first)

1. **Problem:** attention alone doesn't know word **order**. "Dog bites man" = "man bites dog".
2. **sin and cos** describe a point on a circle. They let us **rotate** a vector.
3. **RoPE** = rotate **Q and K** by an angle based on the word's **position**.
4. **Magic:** the Q·K score then depends only on the **distance** between two words, not where they are.
5. **Not V**, only Q and K.
6. **For inference:** K is stored in the KV cache **already rotated**, and **long context** needs **RoPE scaling**.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| RoPE adds a position number to the embedding | It **rotates** Q and K inside every attention layer. |
| RoPE is applied to V too | **Only Q and K.** |
| A model works at any length | Past its trained length the angles are new to it, so quality drops. You need **RoPE scaling**. |

## Why this matters for inference

- Every model you'll serve (Llama, Mistral, Qwen, DeepSeek) uses RoPE.
- Settings like `rope_theta` and `rope_scaling` in `config.json` decide how **long a context** the model can handle.
- The KV cache stores **rotated** K, which matters for prefix caching.

## Real-life example

### Clock hands 🕐

Each word gets a clock hand that turns **30° per position**.

| Word position | Hand angle |
|---|---|
| 3 | 90° |
| 5 | 150° |
| 8 | 240° |
| 10 | 300° |

The **angle between** positions 5 and 3 is 60°, and so is the angle between 10 and 8.

👉 The gap between hands depends only on **how far apart** the words are. That's exactly what RoPE gives attention.

## The math

### 1. sin and cos = a point on a circle

```
angle θ on a circle of size 1 → point (cos θ, sin θ)

θ = 0°   → (1, 0)
θ = 90°  → (0, 1)
θ = 180° → (-1, 0)
```

### 2. Rotating a pair of numbers

```
rotate [x, y] by θ = [x·cos θ − y·sin θ,  x·sin θ + y·cos θ]
```

### 3. RoPE

- Split Q and K into **pairs** of numbers. Llama 3 has a head size of 128, so **64 pairs**.
- Rotate each pair by `position × speed`.
- Each pair has a **different speed**:

| Pair | Speed | One full turn every | Good for |
|---|---|---|---|
| First | fast | ~6 words | nearby words |
| Middle | medium | ~4,400 words | |
| Last | very slow | ~2.5 million words | far-away words |

(Numbers for Llama 3, `rope_theta = 500,000`. A bigger theta = slower pairs = longer reach.)

### 4. Worked example: only distance matters ⭐

q = k = [1, 0], turning 30° per position.

```
Q at 5 → 150° → [-0.87, 0.50]
K at 3 →  90° → [ 0.00, 1.00]
Q·K = 0 + 0.50 = 0.5

Q at 10 → 300° → [ 0.50, -0.87]
K at 8  → 240° → [-0.50, -0.87]
Q·K = -0.25 + 0.75 = 0.5   ✅ same, because both are 2 words apart
```

## Inference connection

### 1. The KV cache stores rotated K

- K is rotated **before** it's saved, so a cached K is tied to its **position**.
- That's why prefix caching works for the **same text at the same position**, e.g. a system prompt at the **start** (KV cache step 14).
- RoPE is cheap maths, so it's **fused** into the Q/K kernel (like 3.3).

### 2. Long context needs RoPE scaling ⭐

The model was trained on, say, 8K words. At word 20,000, the slow pairs reach **angles it has never seen**, and answers get bad.

| Fix | Idea |
|---|---|
| **Position interpolation** | Squeeze positions: word 16,000 is treated like 8,000 (÷2) |
| **NTK / YaRN** | Change the speeds so the slow pairs turn even slower |
| **Llama 3.1 scaling** | Built into the model's config to reach 128K |

In vLLM it's set in the model's `config.json` (`rope_scaling`), and `--max-model-len` sets the limit.

## Remember

- Attention doesn't know order, and **RoPE** fixes it.
- sin and cos = a point on a circle, used to **rotate** pairs of numbers.
- Rotate **Q and K** (not V) by `position × speed`. There are many speeds, from fast to slow.
- Q·K depends only on the **distance** between words.
- The KV cache stores **rotated K**.
- Going past the trained length needs **RoPE scaling** (interpolation, YaRN).

## Quick check

1. Which vectors does RoPE rotate?
2. Q at position 7 and K at 4, vs Q at 103 and K at 100. Same score or different (same q, k)?
3. Why can't a model just run at 4× its trained length with no change?
4. Does a bigger `rope_theta` make the slow pairs turn faster or slower?

<details>
<summary>Answers</summary>

1. **Q and K** (not V).
2. **The same.** Both are 3 positions apart.
3. The slow pairs reach **angles never seen in training**, so quality drops. You need RoPE scaling.
4. **Slower**, so the model can reach further.

</details>
