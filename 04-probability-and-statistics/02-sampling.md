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

## Inference connection

### 1. Sampling has a cost

- Every token, for every user, picks from **~128,000 words** (Llama 3).
- Top-p needs the words **sorted**, and sorting 128K numbers × 256 users every step is slow.
- So engines use special kernels (e.g. FlashInfer's sampling) that **skip the full sort**.

### 2. Penalties

| Setting | What it does |
|---|---|
| `repetition_penalty` | Lowers the score of words already used |
| `presence_penalty` | A fixed penalty if the word has appeared at all |
| `frequency_penalty` | A bigger penalty the more times it has appeared |

These need to know **which words were used before** for each user, which is extra bookkeeping on the server.

### 3. Speculative decoding's accept rule ⭐

A small "draft" model guesses a word, and the big "target" model checks it:

```
Accept the guess with chance = min(1, target chance ÷ draft chance)
```

| Word | Draft chance | Target chance | Accept chance |
|---|---|---|---|
| Paris | 0.80 | 0.60 | 0.60 ÷ 0.80 = **0.75** |
| Lyon | 0.10 | 0.30 | min(1, 3) = **1** |

If a guess is rejected, the server re-samples from the target model's leftover chances. The result is **exactly the same quality** as the big model alone, just faster.

### 4. Temperature 0 isn't always identical

With batching, the same prompt can be in a batch of 1 or 64. That changes the **order of additions**, so tiny rounding differs (2.5). If two words are almost tied, the top word can flip. Same prompt, slightly different answer.

## Remember

- **Greedy** = always the top word (temperature 0).
- **Top-k** = keep k words. **Top-p** = keep words until they add up to p. **Min-p** = keep words ≥ p × the top chance.
- **Re-scale** after trimming, then spin.
- Top-p and min-p **adapt** to how sure the model is.
- Sampling over ~128K words × many users needs **fast kernels**.
- Speculative decoding accepts with **min(1, target ÷ draft)**, so quality is unchanged.

## Quick check

1. Chances: A 0.6, B 0.3, C 0.1. Which words does top-p = 0.85 keep?
2. Same list, top-k = 1. What is that the same as?
3. Same list, min_p = 0.4. What's the cut-off, and which words stay?
4. Draft chance 0.5, target chance 0.2. What's the accept chance?
5. Why can temperature 0 give different answers on a busy server?

<details>
<summary>Answers</summary>

1. 0.6 → 0.9 ≥ 0.85, so **A and B**.
2. **Greedy.**
3. Cut-off = 0.4 × 0.6 = **0.24**, so **A and B** stay (C at 0.1 is cut).
4. 0.2 ÷ 0.5 = **0.4**
5. Different **batch sizes** change the rounding, so near-tied words can **flip**.

</details>
