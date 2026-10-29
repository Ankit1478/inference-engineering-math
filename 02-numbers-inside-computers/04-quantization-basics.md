# 2.4 Quantization Basics: Scale, Round, Dequantize

**Quantization means storing the model's decimal numbers as small whole numbers, so the model gets smaller and faster.**

## Why this matters for inference

- A model is just a long list of numbers, like `0.8, -1.5, 2.0, 0.3, ...`
- These numbers are called **weights**. The model learned them during training.
- A big model has billions of them, and each one takes 2 bytes, so the model is very heavy.
- Quantization makes each number smaller. The model then fits on fewer GPUs and writes words faster.

### Why are some numbers negative?

Each weight says how much something matters:

- **Positive (+)**: "yes, more of this"
- **Negative (−)**: "no, less of this"
- **Near zero**: "doesn't matter much"

Without negative numbers, the model could never say "no".

### Why store whole numbers?

The GPU **can** store decimals. We **choose** whole numbers because they are smaller:

| Type | Example | Space for one number |
|---|---|---|
| Decimal | 0.8 | 2 bytes |
| Small whole number | 5 | 1 byte, or even half a byte |

To write each word, the GPU reads all the numbers in the model. Less to read means faster words.

## Real-life example

### Should I watch this movie?

| Fact | Weight | Why |
|---|---|---|
| Has action | +0.8 | I like it |
| Is 3 hours long | −1.5 | I don't like it |
| Has my favourite actor | +2.0 | I love it |

Total: 0.8 − 1.5 + 2.0 = **1.3**, which is positive, so watch it.

This is exactly what a model does with its weights: positives push the answer up, negatives push it down.

### The suitcase

You *can* pack big clothes, but if you fold them small, more fits and the suitcase is lighter to carry. Quantization is folding the numbers small.

## The math

To keep the maths easy, pretend the box only holds whole numbers from **-10 to 10**. (In real life the box is -127 to 127. It has room for negatives because weights can be negative.)

Numbers: `0.4, -1.0, 2.0`

### Step 1: Scale (work out the size of one step)

Find the biggest number, ignoring the minus sign. Here that is **2.0**.

We want 2.0 to land on the edge of the box, which is 10:

```
scale = 2.0 ÷ 10 = 0.2
```

So each whole number is worth 0.2, like a ruler with a mark every 0.2.

### Step 2: Round (turn each number into a whole number and store it)

| Number | ÷ 0.2 | Stored |
|---|---|---|
| 0.4 | 2 | **2** |
| -1.0 | -5 | **-5** |
| 2.0 | 10 | **10** |

We store `2, -5, 10`, plus the scale `0.2` once.

### Step 3: Dequantize (turn it back when the model reads it)

```
2 × 0.2  = 0.4   ✓
-5 × 0.2 = -1.0  ✓
10 × 0.2 = 2.0   ✓
```

- Rounding happens **once**, when the model is saved.
- Turning it back happens **every time** the model runs.

### The scale is calculated, not picked freely

If you just pick a scale, numbers can overflow. For example, multiplying everything by 100:

```
3.0, -1.5, 0.7  ×100  →  300, -150, 70
```

300 doesn't fit in a box that goes up to 10. It's broken.

So always calculate the scale from the biggest number, so that the biggest number lands exactly on the edge of the box.

Multiplying and dividing are the same idea: dividing by 0.2 is the same as multiplying by 5.

### We lose a little accuracy

Take 0.5 with a scale of 0.2:

```
0.5 ÷ 0.2 = 2.5  →  round  →  3
3 × 0.2   = 0.6
```

We put in **0.5** and got back **0.6**. That small difference is the price we pay for saving memory.

## Inference connection

A 70B model has about 70 billion numbers:

| Stored as | Size | GPUs needed (80 GB each) |
|---|---|---|
| Decimals (2 bytes) | 140 GB | 2 |
| Whole numbers (1 byte) | 70 GB | 1 (tight) |
| Small whole numbers (half a byte) | 35 GB | 1, with room to spare |

Smaller model, so less to read per word, so faster words and cheaper serving.

## Remember

- A model is a list of numbers (weights). Some are positive, some negative.
- We store them as small whole numbers to save memory and go faster.
- `scale = biggest number ÷ biggest whole number allowed`
- `store = round(number ÷ scale)`
- `back = stored × scale`
- The scale is calculated from the biggest number, so nothing overflows.
- We lose a little accuracy (0.5 came back as 0.6).

## Still to learn

- [ ] Why some accuracy gets lost
- [ ] Using fewer bits (INT4)
- [ ] The "one huge number" problem
- [ ] The fix: splitting numbers into small groups

## Quick check

The box holds -10 to 10. Numbers: `4.0, -2.0, 1.0`

1. What is the scale?
2. What whole numbers get stored?
3. Why would multiplying by 100 be a bad choice?

<details>
<summary>Answers</summary>

1. scale = 4.0 ÷ 10 = **0.4**
2. 4.0 ÷ 0.4 = **10**, -2.0 ÷ 0.4 = **-5**, 1.0 ÷ 0.4 = 2.5 → **3** (2 is also acceptable)
3. 4.0 × 100 = 400, which is far past the box edge of 10, so it overflows.

</details>
