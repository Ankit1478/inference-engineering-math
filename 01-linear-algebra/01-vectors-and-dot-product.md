# 1.1 Vectors and Dot Product

**A vector is a list of numbers that describes something, and a dot product multiplies two vectors position by position and adds the results into one number.**

## Why this matters for inference

- Every token (word piece) inside an LLM is a vector of thousands of numbers.
- The dot product is the single smallest operation in an LLM. Matrix multiplication, attention, and choosing the next word are all built from it.
- When you later count FLOPs and estimate speed, you are really counting dot products.

## Real-life example

Imagine describing things with scores from 0 to 1:

| Thing | Sweet | Round | Alive |
|---|---|---|---|
| Apple | 0.9 | 0.8 | 0.1 |
| Dog | 0.0 | 0.2 | 1.0 |

Each row is a vector:

```
apple = [0.9, 0.8, 0.1]
dog   = [0.0, 0.2, 1.0]
```

A vector is just a description written in numbers.

Now imagine you are buying a phone. You care a lot about the camera, a little about battery, and not at all about colour. You give each feature an importance:

```
your priorities = [camera 3, battery 2, color 0]
phone features  = [camera 8, battery 5, color 9]
```

To score the phone, you multiply each feature by its importance and add everything up. That is a dot product.

## The math

### Dot product rule

Multiply matching positions, then add:

```
[a1, a2, a3] · [b1, b2, b3] = a1×b1 + a2×b2 + a3×b3
```

### Worked example 1

```
[1, 2, 3] · [4, 5, 6]

= 1×4 + 2×5 + 3×6
= 4   + 10  + 18
= 32
```

### Worked example 2: the phone score

```
phone      = [8, 5, 9]
priorities = [3, 2, 0]

score = 8×3 + 5×2 + 9×0
      = 24  + 10  + 0
      = 34
```

Notice that the colour was 9 (very good), but its weight was 0, so it was ignored. The weights decide what matters.

### What the weights can do

| Weight | Meaning |
|---|---|
| Big (like 3) | "This feature is very important" |
| 0 | "Ignore this feature" |
| Negative (like -2) | "If this feature is high, lower the score" |

### Worked example 3: "is it a fruit?"

```
apple            = [0.9, 0.8, 0.1]
"fruit detector" = [1,   1,   0  ]

apple · fruit detector = 0.9×1 + 0.8×1 + 0.1×0 = 1.7   ← high

dog              = [0.0, 0.2, 1.0]
dog · fruit detector   = 0.0×1 + 0.2×1 + 1.0×0 = 0.2   ← low
```

A dot product measures **how well a vector matches a pattern**. Big result = strong match.

### Counting the work

For two vectors with `k` numbers each:

- `k` multiplications
- about `k` additions

So a dot product of length `k` costs about `2k` math operations.

## Inference connection

- In a real model, each token becomes a vector of around **4,096 numbers** (for a 7B model). The numbers don't have human labels like "sweet". The model invents its own features during training. The idea is the same.
- One dot product of two 4,096-number vectors is 4,096 multiplications and about 4,096 additions.
- A layer of an LLM does millions of these dot products for every single token. That is why counting them tells you how much work the model does.
- At the very end of the model, the final vector is compared (with dot products) against every word in the vocabulary. The word with the highest score becomes the next word.

```
final vector · "money"   = 0.3
final vector · "water"   = 2.1   ← highest, chosen
final vector · "flowers" = 1.4
```

## Remember

- A vector is a list of numbers that describes something.
- Dot product = multiply position by position, then add.
- Multiply gives each feature its importance. Add combines everything into one score.
- A big dot product means "these two vectors match well".
- A dot product of length `k` costs about `2k` operations.

## Quick check

1. Compute `[2, 0, 1] · [3, 4, 5]`.
2. A judge has weights `[0, 0, 5]`. Which feature does this judge care about?
3. Using the fruit detector `[1, 1, 0]`, score `banana = [0.8, 0.1, 0.0]`.
4. How many multiplications are in a dot product of two vectors with 4,096 numbers?

<details>
<summary>Answers</summary>

1. 2×3 + 0×4 + 1×5 = 6 + 0 + 5 = **11**
2. Only the third feature. The first two have weight 0, so they are ignored.
3. 0.8×1 + 0.1×1 + 0.0×0 = **0.9**, so it is fairly "fruit".
4. **4,096** multiplications (plus about 4,096 additions).

</details>
