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

## Real-life example

### A spinning wheel 🎡

Each word gets a slice of the wheel the size of its chance.

| Method | Wheel |
|---|---|
| Greedy | No spin. Just take the biggest slice. |
| Sampling | Spin the full wheel. |
| Top-k / top-p / min-p | **Cut off the tiny slices first**, then spin. |

The tiny slices are usually junk words ("Pizza" as the capital of France), so cutting them keeps answers sensible but still varied.

## The math

### The example list

After softmax for "The capital of France is …":

| Word | Chance |
|---|---|
| Paris | 0.50 |
| Lyon | 0.20 |
| Nice | 0.16 |
| Rome | 0.09 |
| Pizza | 0.05 |

### 1. Greedy

```
Always "Paris"
```

### 2. Top-k (k = 2)

```
Keep:      Paris 0.50, Lyon 0.20      (sum 0.70)
Re-scale:  Paris 0.71, Lyon 0.29
```

### 3. Top-p (p = 0.8)

Add up from the top until you reach 0.8:

```
Paris 0.50 → 0.50
Lyon  0.20 → 0.70
Nice  0.16 → 0.86 ≥ 0.8, stop

Keep:      Paris, Lyon, Nice           (sum 0.86)
Re-scale:  Paris 0.58, Lyon 0.23, Nice 0.19
```

👉 Top-p **adapts**. If the model is very sure (Paris 0.95), top-p keeps just 1 word. If it's unsure, it keeps many.

### 4. Min-p (min_p = 0.3)

```
Cut-off = 0.3 × top chance = 0.3 × 0.50 = 0.15
Keep:     Paris 0.50, Lyon 0.20, Nice 0.16   (all ≥ 0.15)
```

👉 Also adapts. If the top is 0.95, the cut-off is 0.285, so only Paris is kept.

### 5. The order in vLLM

```
scores → penalties → ÷ temperature → softmax → top-k → top-p / min-p → re-scale → spin
```

### 6. Common settings

| Use | Settings |
|---|---|
| Code, maths, extraction | temperature **0** (greedy) |
| Normal chat | temperature **0.7**, top-p **0.9** |
| Creative writing | temperature **1.0**, top-p **0.95** |

