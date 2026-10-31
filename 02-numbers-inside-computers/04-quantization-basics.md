# 2.4 Quantization Basics: Scale, Round, Dequantize

**Quantization means storing the model's decimal numbers as small whole numbers, so the model gets smaller and faster.**

## Quick summary (read this first)

A model is a list of decimal numbers, some positive and some negative. The GPU *can* store decimals, but we choose whole numbers because they are smaller.

1. Find the biggest number, ignoring the minus sign.
2. **Scale** = biggest ÷ box edge (127 for INT8, 7 for INT4).
3. **Store** = round(number ÷ scale).
4. **Turn back** = stored × scale.
5. You lose a little accuracy. INT4 loses more than INT8.
6. A giant number spoils the scale, so cut the list into groups of 128, each with its own scale.

Example, biggest number 22 in INT8:

```
scale = 22 ÷ 127 = 0.17
5.0 ÷ 0.17 = 29.4 → round → 29      (store 29)
29 × 0.17 = 4.93                    (turn back, close to 5.0)
```

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Divide by any whole number that suits" | Divide by the **box edge**, which the format decides: 127 for INT8, 7 for INT4 |
| "Step 2 is just round" | **Divide by the scale first**, then round |
| "Move the giant number into its own group" | Cut the list into equal groups **in order** (every 128 numbers). The giant stays where it is and only spoils its own group. |

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

### Why we lose accuracy

You already know rounding: `2.7 → 3`. The number moved by 0.3. **That small change is the error.**

Think of money. If you only have **₹2 coins**, you can pay ₹2, ₹4, ₹6… but never ₹5. You are always off a little.

| Price | What you pay | Off by |
|---|---|---|
| ₹4 | ₹4 | ₹0 ✅ |
| ₹5 | ₹4 or ₹6 | ₹1 |
| ₹7 | ₹6 or ₹8 | ₹1 |

**The scale is the coin size.** With scale 0.2 we can only store 0, 0.2, 0.4, 0.6… so 0.5 becomes 0.6:

```
0.5 ÷ 0.2 = 2.5  →  round  →  3
3 × 0.2   = 0.6       (off by 0.1)
```

- **Small coins** (small scale): off by a little ✅
- **Big coins** (big scale): off by a lot ❌

### INT4: a smaller box

| Box | Holds | Size per number | 70B model |
|---|---|---|---|
| INT8 | -127 to 127 | 1 byte | 70 GB |
| INT4 | -7 to 7 | ½ byte | **35 GB** |

INT4 makes the model half the size again. The catch: fewer whole numbers means bigger coins, so bigger errors.

Biggest number 1.4, storing 0.5:

```
INT8:  scale = 1.4 ÷ 127 = 0.011   →  0.5 comes back as 0.496   (off by 0.004) ✅
INT4:  scale = 1.4 ÷ 7   = 0.2     →  0.5 comes back as 0.6     (off by 0.1)   ❌
```

| | INT8 | INT4 |
|---|---|---|
| Size | Bigger | **Smaller** ✅ |
| Speed | Fast | **Faster** ✅ |
| Accuracy | **Very good** ✅ | A bit worse |

Most people still use INT4, because the model usually stays good enough and the savings are huge.

### The "one giant number" problem (outliers)

**Group photo 📸:** you photograph 3 kids, and everyone is clear. Then a **giant** joins. To fit the giant, the camera zooms out a lot, and the kids become tiny dots you can't see.

Quantization is the same:

- **Camera zoom** = the scale
- **Kids** = the many small numbers
- **Giant** = one huge number (called an **outlier**)

The scale must fit the biggest number. One giant number makes it "zoom out", and all the small numbers become **0**:

```
Numbers: 0.3, -0.2, 0.1, 7.0    (INT4)
scale = 7.0 ÷ 7 = 1.0

0.3 → 0    -0.2 → 0    0.1 → 0    7.0 → 7
```

Real models **do** have a few giant numbers among millions of small ones. This is the biggest problem in quantization.

### The fix: separate photos (groups)

Split the numbers into **small groups** (usually 128 numbers each) and give **each group its own scale**.

```
Photo 1: 😀😀😀   → zoomed in, clear ✅
Photo 2: 😀😀😀   → zoomed in, clear ✅
Photo 3: 😀🧍     → zoomed out for the giant
```

The giant only spoils **its own group**. The cost is a tiny bit of extra memory to save one scale per group, and it's worth it. Real tools like **GPTQ** and **AWQ** use this trick.

## Inference connection

A 70B model has about 70 billion numbers:

| Stored as | Size | GPUs needed (80 GB each) |
|---|---|---|
| Decimals (2 bytes) | 140 GB | 2 |
| Whole numbers (1 byte) | 70 GB | 1 (tight) |
| Small whole numbers (half a byte) | 35 GB | 1, with room to spare |

Smaller model, so less to read per word, so faster words and cheaper serving.

## Remember

1. A model is a list of numbers (weights). Some are positive, some negative.
2. We store them as small whole numbers to make the model smaller and faster.
3. `scale = biggest number ÷ box edge` → `store = round(number ÷ scale)` → `back = stored × scale`
4. Rounding loses a little accuracy. Bigger coins (scale) mean more error.
5. INT4 is half the size of INT8, but has bigger errors.
6. One giant number (outlier) makes small numbers become 0. Fix: split into groups, each with its own scale.

## Quick check

1. The box holds -10 to 10. Numbers: `4.0, -2.0, 1.0`. What is the scale, and what whole numbers get stored?
2. You only have ₹5 coins. Something costs ₹12. What do you pay, and how much are you off by?
3. Biggest number is 2.1. What is the scale in INT4 (box edge 7)?
4. Why does one giant number make the small numbers disappear?
5. How do groups fix it?

<details>
<summary>Answers</summary>

1. scale = 4.0 ÷ 10 = **0.4**. Stored: **10, -5, 3** (1.0 ÷ 0.4 = 2.5, so 2 is also acceptable)
2. Pay **₹10**, off by **₹2**
3. 2.1 ÷ 7 = **0.3**
4. The scale comes from the biggest number, so it becomes huge, and small numbers round to 0. Like a camera zooming out for a giant, so the kids become dots.
5. Each group gets its own scale, so a giant only spoils its own group. The other groups stay accurate.

</details>
