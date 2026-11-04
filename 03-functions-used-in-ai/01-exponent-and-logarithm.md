# 3.1 Exponent (eˣ) and logarithm (log)

**eˣ turns any number into a positive number and makes big numbers much bigger. log is the reverse: it turns multiplying into adding.**

## Quick summary (read this first)

1. **Exponent** = multiplying a number by itself many times: 2³ = 2 × 2 × 2 = 8.
2. **e** is a special number, about **2.718**. AI uses **eˣ** a lot.
3. eˣ is **always positive**, and it **grows very fast**.
4. **log** is the reverse of eˣ: `log(eˣ) = x`.
5. log turns **multiplying into adding**: `log(a × b) = log a + log b`.
6. **Where they're used:** eˣ is in **softmax** (next lesson). log is in **logprobs** and **perplexity** (Section 4).

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| e⁻² is negative | **No.** e⁻² = 0.135. eˣ is **always positive**. |
| e⁰ = 0 | **e⁰ = 1.** Anything to the power 0 is 1. |
| The log of a probability is positive | Probabilities are between 0 and 1, so their log is **0 or negative**. |
| log(0) = 0 | log(0) = **minus infinity**. log(**1**) = 0. |

## Why this matters for inference

- Every time the model picks a word, it runs **softmax**, and softmax uses **eˣ**.
- eˣ grows so fast that it can **overflow** FP16 (2.5). Inference code has to handle this.
- APIs and vLLM return **logprobs** (the log of each word's probability). You need log to read them.
- Quality numbers like **perplexity** use log (Section 4).

## Real-life example

### eˣ = a rumour spreading 🗣️

Every day, each person tells 2 more people, so the number keeps multiplying:

| Day | People |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 10 | 1,024 |
| 20 | 1,048,576 |

That's **exponential growth**: slow at first, then huge.

### log = "how many days?" 🔁

log asks the reverse question: **"how many days to reach 1,024 people?"** → 10.

```
2^10 = 1,024   ↔   log₂(1,024) = 10
```

You've seen this already in 2.1: 256 values need **log₂(256) = 8 bits**.

## The math

### 1. eˣ table

| x | eˣ | Note |
|---|---|---|
| -2 | 0.135 | negative x → small, but still positive |
| -1 | 0.368 | |
| 0 | **1** | |
| 1 | 2.718 | that's e |
| 2 | 7.39 | |
| 5 | 148 | |
| 10 | 22,026 | |
| 12 | 162,755 | ❌ too big for FP16 (max 65,504) |

**2 key facts:**
- **Always positive.** Even a very negative score gives a small positive number. Softmax needs this.
- **Grows very fast.** A score of 5 vs 2 is +3, but e⁵ ÷ e² = e³ ≈ **20× bigger**. So eˣ makes the top score **win by a lot**.

### 2. log table (natural log, "ln")

| Number | log | Note |
|---|---|---|
| 1 | **0** | 100% sure |
| 0.5 | -0.69 | |
| 0.2 | -1.61 | |
| 0.1 | -2.30 | |
| 0.01 | -4.61 | |
| 0 | -∞ | impossible |

👉 For probabilities: **closer to 0 = more sure**, **more negative = less sure**.

### 3. The magic rule: multiplying becomes adding

```
log(a × b) = log(a) + log(b)
```

**Worked example:** the model writes 3 words with probabilities 0.5, 0.2 and 0.1.

```
Multiply:  0.5 × 0.2 × 0.1 = 0.01

With log:  -0.69 + -1.61 + -2.30 = -4.61
Back:      e^(-4.61) = 0.01  ✅ same answer
```

**Why bother?** Multiplying many small numbers **underflows to 0** (2.5):

```
50 words, each 0.1:
Multiply:  0.1^50 = 0.000…0001 (50 zeros) → FP32 can't store it → 0 ❌
With log:  50 × -2.30 = -115                → easy to store ✅
```

So computers work with **log probabilities** (logprobs), not raw probabilities.

## Inference connection

| Where | What it uses | Why |
|---|---|---|
| **Softmax** (every word, every layer's attention) | eˣ | Makes all scores positive, so the top score wins clearly |
| **Overflow** | eˣ | e¹² is already too big for FP16. Fix: subtract the biggest score first (next lesson) |
| **Logprobs** in APIs and vLLM | log | Safe to add up, no underflow |
| **Perplexity** (Section 4) | log and eˣ | Measures how surprised the model is. Used to check that quantization didn't hurt quality (2.6) |
| **GPU speed** | eˣ | eˣ is slower than multiplying on a GPU, so FlashAttention is careful how often it computes it |

### Reading logprobs

```
"Paris"  logprob -0.05  → e^(-0.05) = 0.95  very sure ✅
"Lyon"   logprob -3.00  → e^(-3.00) = 0.05  not sure
```

## Remember

- **eˣ** = always positive and grows very fast. e⁰ = 1.
- **log** = the reverse of eˣ. log(1) = 0, and the log of a probability is ≤ 0.
- `log(a × b) = log a + log b`, so multiplying becomes adding with no underflow.
- eˣ is used in **softmax**, log is used in **logprobs** and **perplexity**.
- eˣ can **overflow** FP16 at about x = 11.

## Quick check

1. What is e⁰?
2. Can eˣ ever be negative?
3. Which word is the model surer about: logprob -0.1 or -2.0?
4. Why do computers add logprobs instead of multiplying probabilities?
5. Score A = 3, score B = 1. Roughly how many times bigger is eᴬ than eᴮ?

<details>
<summary>Answers</summary>

1. **1**
2. **No.** It's always positive, even for very negative x.
3. **-0.1.** Closer to 0 = surer (e^(-0.1) ≈ 0.90 vs e^(-2) ≈ 0.14).
4. Multiplying many small probabilities **underflows to 0**. Adding logs doesn't.
5. e³ ÷ e¹ = e² ≈ **7.4×**

</details>
