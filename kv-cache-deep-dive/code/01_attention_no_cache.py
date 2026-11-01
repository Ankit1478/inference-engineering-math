# Step 18: attention WITHOUT a KV cache (the slow way)
# Every round rebuilds K and V for ALL words seen so far.
# Run: pip install torch && python3 01_attention_no_cache.py

import torch

torch.manual_seed(0)

d = 8                        # each word = 8 numbers
W_q = torch.randn(d, d)      # the model's weights for making Q, K, V
W_k = torch.randn(d, d)
W_v = torch.randn(d, d)


def attention(q, K, V):
    scores = q @ K.T / d**0.5                # how well Q matches each K
    weights = torch.softmax(scores, dim=-1)  # turn the scores into percentages
    return weights @ V                       # mix the V's using those percentages


words = torch.randn(10, d)   # 10 words (pretend they're already numbers)
kv_made = 0

for t in range(1, len(words) + 1):
    seen = words[:t]          # all words so far
    K = seen @ W_k            # make K for ALL words again (waste)
    V = seen @ W_v            # make V for ALL words again (waste)
    kv_made += t
    q = seen[-1] @ W_q        # Q only for the newest word
    out = attention(q, K, V)  # a real model would turn this into the next word

print("K,V made:", kv_made)   # 1 + 2 + ... + 10 = 55
