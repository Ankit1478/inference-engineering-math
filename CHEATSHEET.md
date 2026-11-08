# Inference Math Cheatsheet (worked on my Mac)

Every formula from the lessons in one place, with real numbers for **my Mac** running **Llama 3 8B at 4-bit**. To use it for a different model or machine, change the inputs and redo each line.

## 1. My inputs

| What | Value | Where it comes from |
|---|---|---|
| Chip | Apple M5, 10-core GPU, 16 GB | `sysctl -n machdep.cpu.brand_string` |
| Memory speed | **114 GB/s** (Apple says 153) | Measured (script at the bottom) |
| Compute (FP16) | **5.6 TFLOPS** | Measured (script at the bottom) |
| Usable memory | **~12 GB** (75% of 16 GB) | macOS GPU limit |
| Parameters | **8B** | Model name |
| Bytes per weight | **0.5** (4-bit) | 2 = BF16, 1 = 8-bit, 0.5 = 4-bit |
| Layers / KV heads / head size | 32 / 8 / 128 | Model's `config.json` |
| KV cache bytes per number | 2 (BF16) | |

> Real programs reach about **70–85%** of the advertised numbers. My memory test hit 114 of 153 GB/s, which is 75%.

## 2. Does the model fit?

```
Model size     = parameters × bytes per weight
               = 8B × 0.5 = 4 GB  (+ ~0.5 GB extra) ≈ 4.5 GB

Memory needed  = weights + KV cache + ~10% extra
Fits?          = 4.5 GB < 12 GB  ✅
```

| Format | 8B model | 9.4B model |
|---|---|---|
| BF16 (×2) | 16 GB ❌ | 18.8 GB ❌ |
| 8-bit (×1) | 8 GB ⚠️ tight | 9.4 GB ⚠️ tight |
| 4-bit (×0.5) | ~4.5 GB ✅ | ~5.2 GB ✅ |

## 3. KV cache

```
KV per word    = 2 × layers × KV heads × head size × bytes
               = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 0.128 MB

KV per user    = KV per word × words
               = 0.128 MB × 4,000 ≈ 0.5 GB

KV total       = KV per user × users
```

## 4. How many users fit?

```
Users = (usable memory − weights − extra) ÷ KV per user
      = (12 − 4.5 − 1.2) ÷ 0.5 ≈ 12 users  (at 4,000 words each)
```

## 5. Switch point: memory-bound or compute-bound?

```
Switch point = compute ÷ memory speed
             = 5,600 GFLOPS ÷ 114 GB/s ≈ 49
```

- Fewer than ~49 words processed together means **memory-bound**.
- More than ~49 means **compute-bound**.

| Machine | Switch point |
|---|---|
| My Mac (M5) | ~49 |
| H100 | 989 ÷ 3.35 ≈ 295 |
| L4 | 121 ÷ 0.3 ≈ 400 |

## 6. Decode: time per word

```
Memory time   = (weights + KV read) ÷ memory speed
              = (4.5 + 0.128) GB ÷ 114 GB/s ≈ 41 ms

Compute time  = (2 × parameters × users) ÷ compute
              = (2 × 8B × 1) ÷ 5.6T ≈ 3 ms

Real time     = the bigger one = 41 ms   → memory-bound
Words/sec     = 1 ÷ 0.041 ≈ 24 words/s
```

Short version: **words/sec ≈ memory speed ÷ model size**.

## 7. Prefill: time to first word (TTFT)

```
Prefill FLOPs = 2 × parameters × prompt words
              = 2 × 8B × 1,000 = 16 trillion

TTFT          = FLOPs ÷ compute
              = 16T ÷ 5.6T ≈ 2.9 s   → compute-bound
```

The same prompt on an H100 takes about **0.016 s**, roughly 180 times faster.

## 8. Total wait for one answer

```
Latency = TTFT + words out × time per word
        = 2.9 s + 200 × 0.041 s ≈ 11 s
```

## 9. Batching (many users at once)

Example: 8 users, each with 1,000 words of context.

```
Memory time   = (4.5 + 8 × 0.128) ÷ 114 ≈ 48 ms
Compute time  = (2 × 8B × 8) ÷ 5.6T ≈ 23 ms
Real time     = 48 ms   (still memory-bound, since 8 < 49)

Throughput    = users ÷ time per step
              = 8 ÷ 0.048 ≈ 167 words/s   (1 user: 24, about 7× more)
```

## 10. Queuing (people arriving over time)

Example: 1 request every 2 s (0.5/s), each takes 11 s, 8 seats.

```
Little's Law  = arrival rate × time each stays
              = 0.5 × 11 ≈ 5.5 users inside

Service rate  = seats ÷ time each stays = 8 ÷ 11 ≈ 0.73 per second
Busy          = arrival ÷ service = 0.5 ÷ 0.73 ≈ 68%   ✅ (< 100%)

Wait in line  ≈ busy ÷ (1 − busy) × time per seat
              ≈ 0.68 ÷ 0.32 × (11 ÷ 8) ≈ 3 s extra
```

## 11. Quick sheet

```
Size        = params × bytes per weight
KV/word     = 2 × layers × KV heads × head size × bytes
Users       = (memory − weights − extra) ÷ KV per user
Switch      = compute ÷ bandwidth                      (Mac ≈ 49, H100 ≈ 295)
Decode/word = max(bytes read ÷ bandwidth, 2 × params × users ÷ compute)
Words/sec   ≈ bandwidth ÷ model size                   (1 user)
TTFT        = 2 × params × prompt words ÷ compute
Latency     = TTFT + words out × time per word
Throughput  = users ÷ time per step
Little      = users inside = arrival rate × time each stays
Busy        = arrivals ÷ (seats ÷ time each)           (must stay < 100%)
Wait        ≈ busy ÷ (1 − busy) × time per seat
GPUs (load) = arrivals ÷ (one GPU's capacity × ~0.75), rounded up
```

## 12. Measure your own machine

```bash
python3 -m venv venv && ./venv/bin/pip install mlx
./venv/bin/python bench.py
```

`bench.py`:

```python
import mlx.core as mx, time

# Compute: big FP16 matrix multiply
n = 8192
a = mx.random.normal((n, n)).astype(mx.float16)
b = mx.random.normal((n, n)).astype(mx.float16)
mx.eval(a, b)
for _ in range(3): mx.eval(a @ b)          # warm up
t = time.time(); R = 10
for _ in range(R): mx.eval(a @ b)
dt = (time.time() - t) / R
print(f"Compute: {2 * n**3 / dt / 1e12:.1f} TFLOPS (FP16)")

# Memory: read + write a 2 GB array
x = mx.zeros((512 * 1024 * 1024,), dtype=mx.float32)
mx.eval(x)
for _ in range(3): mx.eval(x + 1)          # warm up
t = time.time(); R = 10
for _ in range(R): mx.eval(x + 1)
dt = (time.time() - t) / R
print(f"Memory:  {2 * x.nbytes / dt / 1e9:.0f} GB/s")
```

My result: **5.6 TFLOPS** and **114 GB/s**.

## 13. Check the maths against reality

When the model runs, compare:

| | Predicted | Measured |
|---|---|---|
| Decode speed (1 user) | ~24 words/s | _to fill_ |
| TTFT (1,000-word prompt) | ~2.9 s | _to fill_ |
| Bandwidth actually used = size × words/s | ~108 GB/s | _to fill_ |
