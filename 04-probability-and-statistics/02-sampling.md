# 4.2 Sampling: temperature, top-k, top-p ⭐

**Softmax gives a probability for every word. Sampling is how one word actually gets picked: always the top one, or a random pick from a trimmed list.**

## Quick summary (read this first)

1. **Greedy** (temperature 0): always pick the top word. Same input → same output.
2. **Sampling**: pick randomly **by probability**, like spinning a wheel.
3. **Temperature**: sharpen or flatten the probabilities first (3.2).
4. **Top-k**: keep only the **k** best words.
5. **Top-p**: keep the best words until their chances **add up to p**.
6. **Min-p**: keep words that are at least **p × the top word's chance**.
7. After trimming, **re-scale** so the kept chances add up to 1 again, then spin.
8. **For inference:** sampling runs for **every user, every token**, over ~128K words, so it needs fast kernels. Speculative decoding has its own **accept rule**.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| Temperature 0 = random | Temperature 0 = **greedy**, always the top word. |
| Top-p 0.9 = keep the top 90% of words | It keeps the words whose **chances add up to 90%**, which might be only 2 words. |
| Same seed + temperature 0 = always identical output on a server | Not guaranteed. **Batching** can change tiny rounding, so the top word can flip. |
| Trimmed chances are used as they are | They're **re-scaled** to add up to 1. |

## Why this matters for inference

- Every request sets these in vLLM's `SamplingParams` (`temperature`, `top_k`, `top_p`, `min_p`, `seed`).
- Wrong settings look like a "bad model": repeating itself (too greedy) or talking nonsense (too random).
- Sampling over the whole vocabulary for every user is real GPU work.
- **Speculative decoding** must sample in a special way so the output stays exactly as good.

