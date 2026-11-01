# Step 19: attention WITH a KV cache, compared to WITHOUT
# Run: pip install torch && python3 02_attention_with_cache.py

import time

import torch

torch.manual_seed(0)
d = 64
W_q, W_k, W_v = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d)


def attention(q, K, V):
    weights = torch.softmax(q @ K.T / d**0.5, dim=-1)
    return weights @ V


# Without a cache (from step 18)
def no_cache(words):
    outs = []
    for t in range(1, len(words) + 1):
        seen = words[:t]
        K = seen @ W_k                  # rebuild K for ALL words
        V = seen @ W_v                  # rebuild V for ALL words
        q = seen[-1] @ W_q
        outs.append(attention(q, K, V))
    return torch.stack(outs)


# With a cache
def with_cache(words):
    n = len(words)
    K_cache = torch.zeros(n, d)         # the KV cache (reserved for all words up front)
    V_cache = torch.zeros(n, d)
    outs = []
    for t in range(n):
        x = words[t]                    # only the NEW word
        K_cache[t] = x @ W_k            # make its K once, save it
        V_cache[t] = x @ W_v            # make its V once, save it
        q = x @ W_q
        outs.append(attention(q, K_cache[:t + 1], V_cache[:t + 1]))
    return torch.stack(outs)


# 1. Same answer?
words = torch.randn(2000, d)
a = no_cache(words)
b = with_cache(words)
print("First 4 numbers of the output for word 2,000:")
print("  Without cache:", [round(v, 4) for v in a[-1, :4].tolist()])
print("  With cache:   ", [round(v, 4) for v in b[-1, :4].tolist()])
print("  Biggest difference anywhere:", f"{(a - b).abs().max().item():.1e}")
print("  Same answer?", torch.allclose(a, b, atol=1e-3))
print()

# 2. Which is faster?
print(f"{'Words':>6} | {'Without':>8} | {'With':>8} | Faster by")
for n in [500, 1000, 2000, 4000]:
    words = torch.randn(n, d)
    start = time.time(); no_cache(words);   t1 = time.time() - start
    start = time.time(); with_cache(words); t2 = time.time() - start
    print(f"{n:>6} | {t1:>7.2f}s | {t2:>7.2f}s | {t1 / t2:.1f}×")
