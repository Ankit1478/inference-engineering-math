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

