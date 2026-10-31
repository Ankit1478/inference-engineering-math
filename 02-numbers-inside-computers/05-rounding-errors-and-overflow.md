# 2.5 Rounding Errors and Overflow

**A rounding error is when a number is stored a little off; overflow is when a number is too big for its box and becomes completely wrong.**

## Quick summary (read this first)

1. **Rounding error**: the number is a **little** off. Many small errors **add up**.
2. **Overflow**: the number is **too big for its box**, so it becomes **completely** wrong.
3. **Fixes**: add in a bigger box (FP32), use BF16 instead of FP16, and subtract the biggest score in softmax.

| | Rounding error | Overflow |
|---|---|---|
| What happens | Number is a **little** off | Number is **completely** wrong |
| Example | 0.5 → 0.6 | 150 → -106 |
| Cause | Not enough detail | Not enough room |
| Fix | Add in a bigger box | Bigger box, BF16, or the right scale |

### Common mistakes

| ❌ Wrong | ✅ Right |
|---|---|
| "Both problems are fixed by adding in a bigger box" | A bigger box fixes **adding many numbers**. For overflow you may also need **BF16** or a trick like subtracting the biggest score. |
| "T5 gave NaN because FP16 saves memory" | FP16 was chosen **to save memory**. The NaN came from **overflow**: numbers grew past 65,504. |

## Why this matters for inference

- A model does billions of additions per word. Tiny errors can pile up.
- If a number overflows, the model outputs `NaN` and garbage text.
- These two problems decide which number formats (FP16, BF16, FP32, INT8) are safe to use, and where.

## Real-life example

### Rounding: the shop bill

A shop rounds every item to the nearest ₹1:

| Item | Real price | Rounded |
|---|---|---|
| Pen | ₹10.40 | ₹10 |
| Pencil | ₹5.40 | ₹5 |
| Eraser | ₹3.40 | ₹3 |

Each item is only off by ₹0.40. On **1,000 items**, you'd be off by **₹400**.

### Overflow: the car's km counter

An old car's km counter with only 3 digits:

```
997 → 998 → 999 → 000 ❌
```

After 999 there's no room, so it wraps back to 000. The car didn't go back to zero km. The counter just ran out of space.

## The math

### Step 1: Rounding errors

Computers can't store every decimal exactly. In Python:

```python
0.1 + 0.2
# 0.30000000000000004
```

**Why?** Computers store numbers in binary (1s and 0s). Some decimals, like 0.1, can't be written exactly in binary, just like 1 ÷ 3 = 0.3333… never ends. So the computer rounds a tiny bit.

**One error is tiny. Many errors add up.** A dot product inside a model adds about **8,000 numbers**. If every addition is a little off, the final answer drifts.

### Step 2: Overflow

**INT8** holds -128 to 127:

```
100 + 50 = 150   → doesn't fit → wraps around to -106 ❌
```

No error message. A big positive number quietly becomes a negative one.

**FP16** has a biggest number of **65,504**:

```
60,000 + 10,000 = 70,000  → too big → inf → then NaN ❌
```

`inf` means infinity. `NaN` means "not a number". Once a `NaN` appears, everything after it becomes `NaN` too.

### Step 3: The fixes

| Problem | Fix |
|---|---|
| Adding many small numbers | Add in a **bigger box** (FP32), round only once at the end |
| Adding many INT8 numbers | Add in **INT32** (goes up to about 2 billion) |
| FP16 too small for big numbers | Use **BF16**, which is the same 2 bytes but has a huge range |
| A number past the box edge in quantization | Choose the scale from the biggest number, so nothing goes past the edge (from 2.4) |

## Inference connection: real AI examples

### Example 1: Softmax overflow, inside every LLM

To pick the next word, the model gives every word a **score**, then **softmax** turns the scores into chances (Section 3). Softmax uses **eˣ**, which grows extremely fast:

```
e^5  = 148
e^10 = 22,026
e^12 = 162,754   ← bigger than FP16's limit of 65,504!
```

One score of 12 can give `inf`, then `NaN`, then garbage.

**The fix every model uses:** subtract the biggest score first.

```
scores:          10, 11, 12
minus biggest:   -2, -1,  0    → e^0 = 1, nothing explodes ✅
```

The answer is exactly the same, but the numbers stay small.

### Example 2: The T5 model breaking in FP16

**T5** is a well-known Google model. People ran it in FP16 to save memory, but some numbers inside grew bigger than 65,504. FP16 overflowed, and the model output `NaN`.

**The fix:** run it in **BF16**, which has the same 2 bytes but a huge range. This is one big reason BF16 became the standard for LLMs.

### Example 3: BF16 rounding, where adding stops working

BF16 has a big range but **little detail**. Near 256, it can only store every 2nd number: 256, 258, 260…

```
256 + 1 = 257  → no box for 257 → rounds back to 256
```

Add 1 to 256 **a thousand times** in BF16, and the answer is still **256**. The true answer is 1,256.

**The fix:** GPUs add in **FP32**, then round to BF16 once at the end.

## Remember

1. Rounding error = a **little** off. Many small errors **add up**.
2. Overflow = **too big for the box**, so the number becomes **completely** wrong.
3. INT8: 100 + 50 = -106. FP16: above 65,504 gives `inf` then `NaN`.
4. Add in a bigger box (FP32 / INT32) and round only once at the end.
5. Use BF16 instead of FP16 to avoid overflow.
6. Softmax subtracts the biggest score first, so eˣ never explodes.

## Quick check

1. A shop is off by ₹0.50 on every item. You buy 200 items. Worst case, how much is the total off by?
2. A 2-digit counter shows **98**. You add **5**. What does it show?
3. Why does T5 work in BF16 but break in FP16?
4. In BF16, what is 256 + 1?
5. Scores are `20, 21, 22`. What do they become after subtracting the biggest?

<details>
<summary>Answers</summary>

1. 200 × ₹0.50 = **₹100**
2. **03** (it wraps around after 99)
3. Numbers inside T5 grow past 65,504, so FP16 **overflows** and gives `NaN`. BF16 has a huge range, so nothing overflows.
4. **256**. There is no box for 257 in BF16, so it rounds back.
5. **-2, -1, 0**

</details>
