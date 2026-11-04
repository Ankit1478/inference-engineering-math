# 5.2 Attention formula: softmax(QKᵀ / √d) × V ⭐

**Attention lets every token look back at earlier tokens and take a weighted mix of their information. The formula is 3 steps: score (QKᵀ), turn the scores into weights (softmax), mix (× V).**

## Quick summary (read this first)

1. **Q** = what I'm looking for. **K** = what each token offers. **V** = the information each token carries (KV cache step 2).
2. **Step 1, score:** `Q × Kᵀ` = a dot product (1.1) between my question and every token's label.
3. **Step 2, scale:** divide by **√d** (d = head size, 128 → √128 ≈ 11.3) so the scores don't get too big.
4. **Step 3, mask:** future tokens get **−∞**, so they get weight 0 (causal mask).
5. **Step 4, softmax** (3.2): scores → weights that add up to 1.
6. **Step 5, mix:** `weights × V` = a weighted average of the information.
7. **For inference:** in decode Q is **1 row**, so the cost grows with **context length**. In prefill the score matrix is **n × n**, which is why FlashAttention exists.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| √d is there for accuracy | It's there so softmax doesn't become **all-or-nothing**. |
| Attention cost is fixed per token | In decode it grows with **context length** (reading the whole KV cache). |
| Prefill stores the full n × n score matrix | **FlashAttention** never stores it: it uses chunks plus online softmax (3.2). |
| The mask is applied after softmax | It's applied **before**: −∞ → e^(−∞) = 0. |

