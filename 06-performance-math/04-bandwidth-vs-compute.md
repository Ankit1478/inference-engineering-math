# 6.4 Memory bandwidth vs compute speed

**A GPU has two speeds: how fast it reads memory and how fast it does maths. Every job waits for whichever is slower. Decode waits for memory, and prefill waits for maths.**

Example throughout: **Llama 3 8B on an H100**.

| Step | Idea |
|---|---|
| 1 | A GPU has **two speeds** |
| 2 | Two times: the **bigger** one wins |
| 3 | Decode vs prefill |
| 4 | The **switch point** (~500) |
| 5 | Which **fix** for which problem |

## Step 1: A GPU has two speeds

Remember the student story 📖: **reading** the book and **thinking**.

| Speed | Measures | Unit | H100 (round number) |
|---|---|---|---|
| 🚚 **Bandwidth** | reading memory | GB/s | **2,000 GB/s** |
| 👨‍🍳 **Compute** | doing maths | FLOPS | **1,000 TFLOPS** |

```
Reading:  2 trillion bytes per second
Maths:    1,000 trillion FLOPs per second

Maths ÷ reading = 1,000 ÷ 2 = 500
```

👉 The GPU can do **500 maths actions** in the time it reads **1 byte**. To keep the chef busy, every byte delivered must be used about 500 times.

(The real H100 is ~3,350 GB/s and ~990 TFLOPS, a ratio of ~300. We use round numbers.)

## Step 2: Two times. The bigger one wins

```
Memory time = bytes ÷ 2,000 GB/s
Maths time  = FLOPs ÷ 1,000 TFLOPS
Real time   ≈ the BIGGER of the two
```

🍕 **Pizza order:** if the dough takes 8 minutes to arrive and baking takes 1 second, you wait **8 minutes**. A faster oven doesn't help.

