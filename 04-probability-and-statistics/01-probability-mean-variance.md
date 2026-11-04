# 4.1 Probability, mean and variance

**Probability says how likely something is. Mean and variance describe a list of numbers: where its middle is and how spread out it is. RMSNorm in every Llama layer is built on this.**

## Quick summary (read this first)

1. **Probability** = a number from **0** (never) to **1** (always). All options **add up to 1**.
2. **AND** (both happen) → **multiply**. **OR** (one or the other) → **add**.
3. **Expected value** = the average result if you repeat many times.
4. **Mean** = the average. **Variance** = how spread out the numbers are. **Std** = √variance.
5. **RMSNorm** = divide a vector by its **RMS** (root mean square) so its size stays steady. Llama uses it in every layer.
6. **For inference:** RMSNorm is **fused** and kept in **high precision**, and **outliers** (far from the mean) are what make quantization hard.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| A probability can be 1.5 or -0.2 | Always **between 0 and 1**. |
| Chance of A **and** B = A + B | **Multiply**: 0.9 × 0.8 = 0.72. |
| Variance and std are the same | Std = **√variance**. Std is in the same units as the data. |
| RMSNorm = LayerNorm | RMSNorm **skips subtracting the mean**, so it's cheaper. |

## Why this matters for inference

- The model's output is a **probability list** over the next word (softmax, 3.2).
- **AND = multiply** explains logprobs (3.1) and speculative decoding.
- **Expected value** is used for capacity planning: "how many tokens will an average answer use?"
- **RMSNorm** runs twice per layer, so it's a hot kernel to optimise.
- **Std and outliers** are the reason INT8 activations are hard (2.7).

## Real-life example

### Rolling a die 🎲

| Question | Rule | Answer |
|---|---|---|
| Chance of a 6? | 1 of 6 | 1/6 ≈ 0.17 |
| All 6 options together? | Add up to 1 | 6 × 1/6 = 1 ✅ |
| A 6 **and** another 6 (2 rolls)? | Multiply | 1/6 × 1/6 = 1/36 |
| A 5 **or** a 6? | Add | 1/6 + 1/6 = 1/3 |
| Average roll (expected value)? | Average of results | (1+2+3+4+5+6) ÷ 6 = **3.5** |

