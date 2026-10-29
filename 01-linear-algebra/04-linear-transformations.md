# 1.4 Linear Transformations (and the ReLU Filter)

**A matrix is a machine that changes a vector (stretch, shrink, rotate, flip), and because stacked multiplications collapse into one, LLMs need a small "filter" between them.**

## Why this matters for inference

- Every `W × vector` you compute is a linear transformation. You already know how to calculate it. This lesson adds the intuition.
- **RoPE**, the way most modern LLMs encode word order, is a rotation, which is a linear transformation (Section 5).
- It explains why each feed-forward block has an activation function like ReLU between its two matrices.
- Honest importance: medium-low. Know the idea; you won't compute these by hand.

## Real-life example

### Photo editing

On your phone you can zoom, stretch, or rotate a photo. Each of those edits moves every point of the photo using a small matrix. That is a linear transformation.

### Two discounts

A shop gives 10% off, then another 20% off:

```
₹100 → 10% off → ₹90 → 20% off → ₹72
```

You get the same result with one 28% discount:

```
₹100 → 28% off → ₹72
```

Two simple steps were replaced by one. The second step added nothing new. Layers that only multiply behave the same way.

## The math

### Stretch

```
| 2  0 |     | 1 |     | 2×1 + 0×1 |     | 2 |
| 0  3 |  ×  | 1 |  =  | 0×1 + 3×1 |  =  | 3 |
```

`[1, 1]` became `[2, 3]`: x was stretched 2×, y was stretched 3×.

### Rotate by 90°

```
| 0  -1 |     | 1 |     | 0×1 + (-1)×0 |     | 0 |
| 1   0 |  ×  | 0 |  =  | 1×1 +   0×0  |  =  | 1 |
```

`[1, 0]` (pointing right) became `[0, 1]` (pointing up). The length is still 1. **Rotation changes direction, not size.**

Another one: `[3, 4]` rotated 90° becomes `[-4, 3]`. Both have length 5.

### What "linear" means

- Double the input → the output doubles.
- Add two inputs → the outputs add.

No curves, no "if" rules.

### Why stacked linear layers collapse

Use the simplest possible "layer": multiply by a number.

```
Layer 1: × 2
Layer 2: × 3

5 → ×2 → 10 → ×3 → 30
Same as one layer of × 6:   5 → ×6 → 30
```

With matrices it is the same:

```
W2 × (W1 × x) = (W2 × W1) × x = one matrix × x
```

So 32 layers that only multiply would be no smarter than 1 layer.

### The fix: a filter between layers (ReLU)

ReLU is a tiny rule:

```
If the number is negative → make it 0
If the number is positive → keep it
```

```
-5 → 0
-2 → 0
 0 → 0
 3 → 3
 7 → 7
```

Now try Layer 1 = × 2, then ReLU, then Layer 2 = × 3:

```
Input  5:   5 → 10 → (positive, keep) 10 → 30
Input -5:  -5 → -10 → (negative → 0)   0 →  0
```

| Input | Without filter (just × 6) | With filter |
|---|---|---|
| 5 | 30 | 30 |
| -5 | -30 | 0 |

No single multiplication gives "30 for 5 but 0 for -5". The filter added a **decision**, so the layers can no longer collapse into one. Every layer can now add something new.

Large LLMs use smoother versions of ReLU called **GELU** or **SiLU** (Section 3). The idea is the same: keep useful signals, suppress the rest.

## Inference connection

### Where the filter sits

The filter lives inside the **feed-forward** part of every layer. Feed-forward is not a transfer between layers; it is the second half of each layer, after attention:

```
vector → × W1 → ReLU → × W2 → output
```

- The filter has **no weights**, so nothing extra is read from memory.
- It is one tiny check per number, so it is very cheap.
- It is **not** what makes inference slow. Matrix multiplication and memory reads are.

### RoPE (preview)

- The model needs word order: "dog bites man" ≠ "man bites dog".
- RoPE rotates each word's Q and K vectors by an angle that depends on the word's position.
- Rotation is used because it **doesn't change the vector's size**. It only adds position information.

## Remember

- A linear transformation = a matrix changing a vector (stretch, rotate, flip). Every `W × x` is one.
- Rotation changes direction, not length. RoPE uses this.
- Multiply after multiply = one multiply, so stacking only-linear layers is useless.
- The filter (activation function, e.g. ReLU) adds an "if" decision between the two matrices of feed-forward.
- The filter is cheap and has no weights. It is not an inference bottleneck.

## Quick check

1. Rotate `[0, 1]` by 90° using `| 0 -1 |`, `| 1 0 |`. What do you get?
2. Layer 1 = × 4, then ReLU, then Layer 2 = × 2. What is the output for input 3? For input -3?
3. Why can't we stack only linear layers without a filter?

<details>
<summary>Answers</summary>

1. Row 1: 0×0 + (-1)×1 = -1. Row 2: 1×0 + 0×1 = 0. Answer: **[-1, 0]** (now pointing left).
2. Input 3: 3 → 12 → keep 12 → **24**. Input -3: -3 → -12 → 0 → **0**.
3. Stacked linear layers collapse into a single matrix, so the extra layers would add nothing.

</details>
