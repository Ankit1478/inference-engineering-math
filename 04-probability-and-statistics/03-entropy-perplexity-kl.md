# 4.3 Entropy, perplexity and KL divergence

**These are scores for checking a model. Perplexity says how confused the model is by real text. KL says how different two models' answers are. Inference engineers use them to check that quantization didn't hurt the model.**

## Quick summary (read this first)

1. **Surprise** of a word = `−log(chance)`. A sure word gives 0 surprise. A rare word gives a big surprise.
2. **Entropy** = how **unsure** the model is (its average surprise over its own list).
3. **Cross-entropy** = the average surprise at the **real** next words of a text.
4. **Perplexity** = `e^(cross-entropy)` = "how many words is the model confused between?" **Lower is better.**
5. **KL divergence** = how **different** two chance lists are. **0 = identical.**
6. **For inference:** compare the original vs the quantized model. Perplexity and KL should barely change.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| Higher perplexity = better | **Lower** is better. |
| Perplexity 6 means 6% wrong | It means "as confused as choosing between about **6 words**". |
| KL can be negative | KL is always **0 or more**. 0 = the same. |
| Perplexity is all you need | Also run **real tasks** (maths, code, long answers). Perplexity can hide problems. |

## Why this matters for inference

- After FP8 or INT4 (2.6, 2.7), you need a **number** that says "still good" or "broken".
- Perplexity and KL are quick to compute, with no human checking needed.
- Tools like **llama.cpp** report both when you quantize.
- In speculative decoding, a draft model with **low KL** to the target gets more guesses accepted.

## Real-life example

### A quiz contestant 🎤

| Situation | Feeling | Score |
|---|---|---|
| Sure of the answer, and it's right | Not surprised | Low surprise ✅ |
| Gave the right answer only a small chance | Very surprised | High surprise ❌ |
| Over a whole quiz | Average surprise | **Cross-entropy** |
| "It's like I was guessing between ~2 options" | | **Perplexity = 2** |

## The math

### 1. Surprise = −log(chance)

| Chance of the real word | Surprise |
|---|---|
| 1.0 | 0 (no surprise) |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 (very surprised) |

(This is just the log table from 3.1 with the minus sign removed.)

### 2. Entropy = how unsure the model is

| Model's list | Entropy | Meaning |
|---|---|---|
| [1.0] | 0 | totally sure |
| [0.9, 0.05, 0.05] | 0.39 | quite sure |
| [0.5, 0.5] | 0.69 | torn between 2 |
| [0.25, 0.25, 0.25, 0.25] | 1.39 | torn between 4 |

### 3. Cross-entropy and perplexity ⭐

Give the model a real text. For each word, check what chance it gave **the real next word**:

```
Chances it gave the real words:  0.5   0.2   0.8   0.4
Surprise (−log):                 0.69  1.61  0.22  0.92

Cross-entropy = average = 0.86
Perplexity    = e^0.86  = 2.36
```

👉 "The model is about as confused as choosing between **2.4 words**."

Quick sense check: if the model spread its chance evenly over 4 words, perplexity = **4**.

### 4. KL divergence: how different two lists are

```
KL(P ‖ Q) = sum of  P × log(P ÷ Q)
```

| Word | Original (P) | Quantized (Q) |
|---|---|---|
| Paris | 0.7 | 0.6 |
| Lyon | 0.2 | 0.3 |
| Nice | 0.1 | 0.1 |

```
0.7 × log(0.7/0.6) = +0.108
0.2 × log(0.2/0.3) = −0.081
0.1 × log(0.1/0.1) =  0
KL = 0.027   → very close ✅
```

👉 **0 = identical.** The bigger it is, the more the quantized model has changed.

