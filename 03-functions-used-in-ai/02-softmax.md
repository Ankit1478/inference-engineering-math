# 3.2 Softmax ⭐

**Softmax turns any list of scores into probabilities that are all positive and add up to 1, and the biggest score gets the biggest share.**

## Quick summary (read this first)

1. **Input:** any scores, which can be negative or huge. **Output:** probabilities (0 to 1) that **add up to 1**.
2. **Formula:** `softmax(xᵢ) = eˣⁱ ÷ (sum of all eˣ)`
3. It uses **eˣ** (3.1), so every output is positive and the top score **wins by a lot**.
4. **Safe trick:** subtract the biggest score first. Same answer, no overflow.
5. **Online softmax:** you can do it in chunks with a running max and running sum. This is the trick **FlashAttention** is built on.
6. **Used in 2 places:** inside **attention** (every layer) and at the **end** to pick the next word.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| Softmax = divide each score by the total | That breaks with negative scores. Softmax does **eˣ first**, then divides. |
| Subtracting the max changes the answer | **Same answer.** It only avoids overflow. |
| Softmax picks the word | It only gives **probabilities**. Sampling picks the word (Section 4). |

## Why this matters for inference

- Softmax runs in **every attention layer** for **every token**, so it must be fast and safe.
- Without the safe trick, FP16 **overflows** at a score of about 11 (3.1).
- **FlashAttention** exists because of a softmax trick (online softmax).
- **Temperature** (a vLLM setting) is just a change to softmax's input.

## Real-life example

### Sharing a prize 🏆

3 players scored 2, 1 and 0. A ₹100 prize is shared, but the top scorer should get much more.

| Player | Score | eˣ | Share |
|---|---|---|---|
| A | 2 | 7.39 | **₹66.5** |
| B | 1 | 2.72 | ₹24.5 |
| C | 0 | 1.00 | ₹9.0 |
| | | sum = 11.11 | ₹100 ✅ |

A beat B by 1 point but got **2.7× more**. That's eˣ at work.

## The math

### 1. The formula

```
softmax(xᵢ) = eˣⁱ ÷ (e^x₁ + e^x₂ + ... + e^xₙ)
```

**Worked example:** scores `[2, 1, 0]`

```
eˣ:     [7.39, 2.72, 1.00]     sum = 11.11
÷ sum:  [0.665, 0.245, 0.090]  adds up to 1 ✅
```

### 2. The safe trick: subtract the max ⭐

Scores `[1000, 999]` → e¹⁰⁰⁰ = overflow ❌

Subtract the max (1000) first:

```
[1000, 999] → [0, -1] → eˣ = [1, 0.37] → [0.73, 0.27] ✅
```

**Why is it the same answer?** Subtracting a number from every score multiplies the top and the bottom of the fraction by the same amount, so it cancels out. Every real softmax does this.

### 3. Online softmax: in chunks (FlashAttention's trick) ⭐

Normal softmax needs **all** the scores first (to find the max and the sum). For a long chat, that's a huge row of scores sitting in slow GPU memory.

**Online softmax** reads the scores in chunks and keeps just 2 numbers: the **running max** and the **running sum**.

Scores: `[2, 1]` then `[3, 0]`

```
Chunk 1 [2, 1]:  max = 2   sum = e⁰ + e⁻¹ = 1.368

Chunk 2 [3, 0]:  new max = 3 (bigger!)
                 fix the old sum: 1.368 × e^(2−3) = 0.503
                 add new:         e⁰ + e⁻³        = 1.050
                 sum = 1.553

All at once:     e⁻¹ + e⁻² + e⁰ + e⁻³ = 1.553 ✅ same
```

👉 When a bigger max shows up, just **rescale** the old sum. So the GPU never needs the whole row at once. It works chunk by chunk in **fast on-chip memory**. That's FlashAttention.

### 4. Temperature: divide the scores first

```
softmax(scores ÷ T)
```

Scores `[2, 1, 0]`:

| Temperature | Probabilities | Feel |
|---|---|---|
| 0.5 | [0.87, 0.12, 0.02] | sharp, safe, repetitive |
| **1** | [0.67, 0.25, 0.09] | normal |
| 2 | [0.51, 0.31, 0.19] | flat, creative, random |
| → 0 | [1, 0, 0] | always the top word (greedy) |

## Inference connection

| Where | Softmax over | Size |
|---|---|---|
| **Attention** | scores of the new word vs every old word | one row per head per layer, grows with chat length |
| **End of the model** | scores (logits) for every word in the vocabulary | about 128,000 words (Llama 3) |

- **Safe trick** → no FP16 overflow.
- **Online softmax** → FlashAttention, no giant score matrix in memory.
- **Temperature** → the `temperature` setting in vLLM and APIs.

## Remember

- Softmax = **eˣ, then divide by the sum** → positive, adds up to 1.
- The top score wins **by a lot** because of eˣ.
- **Subtract the max** first: same answer, no overflow.
- **Online softmax** = running max + running sum, in chunks → FlashAttention.
- **Temperature** divides the scores: low = sharp, high = flat.

## Quick check

1. Softmax of `[5, 5]`?
2. Why subtract the max before eˣ?
3. Which 2 numbers does online softmax keep?
4. Temperature 0.1 or 1.5: which gives more random answers?
5. Is the output of softmax the chosen word?

<details>
<summary>Answers</summary>

1. **[0.5, 0.5].** Equal scores get equal shares.
2. To avoid **overflow**. The answer doesn't change.
3. The **running max** and the **running sum**.
4. **1.5.** A higher temperature gives flatter probabilities.
5. **No.** It gives probabilities. Sampling picks the word.

</details>
