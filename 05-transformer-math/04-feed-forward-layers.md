# 5.4 Feed-forward layers (FFN) and MoE

**After attention mixes information between tokens, the FFN processes each token on its own: make it wider, filter it, shrink it back. It holds most of the model's weights. MoE models keep many FFNs and use only a few per token.**

## Quick summary (read this first)

1. **Attention** = tokens talk to each other. **FFN** = each token thinks **alone**.
2. Llama's FFN is **SwiGLU** (3.3): `Down( SiLU(Gate·x) × Up·x )`, so **3 matrices**.
3. It goes **wide** (4,096 → 14,336) and then **back** (14,336 → 4,096).
4. FFN = **176M parameters per layer** → **5.6B** in Llama 3 8B (**~70%**).
5. **FLOPs per token = 2 × FFN parameters** (1.3).
6. **MoE** (Mixture of Experts): many FFNs ("experts"), and a **router** picks a few per token. **All** must be in memory, but only a few are **computed**.
7. **For inference:** the FFN is the biggest part to **read** (decode), **quantize** and **split** across GPUs.

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| Attention is the biggest part of the model | The **FFN** is ~70% of the weights. |
| The FFN looks at other tokens | Each token goes through it **alone**. |
| MoE 47B with 13B active needs memory for 13B | It needs memory for **all 47B**. Only the **compute** is 13B. |
| FFN cost grows with chat length | It doesn't. That's attention (5.2). |

