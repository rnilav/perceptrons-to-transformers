# Attention Mechanisms: Mathematical Deep Dive

A companion to [Blog Post #9: Attention Mechanisms](https://dev.to/rnilav/attention-mechanisms-stop-compressing-start-looking-back-1bol).

This document covers the full mathematical treatment of every concept introduced in the blog post — dot-product attention, scaled dot-product attention, the Query/Key/Value framework, self-attention, and multi-head attention. Each section includes variable definitions, step-by-step derivations, worked numerical examples, and tuning guidance.

---

## 1. The RNN Encoder-Decoder Bottleneck

Before attention, the standard sequence-to-sequence architecture compressed the entire input into a single vector.

### The Setup

An RNN encoder reads an input sequence `x₁, x₂, ..., xₙ` and produces hidden states at each step:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**Variables:**
- `x_t ∈ ℝᵈ` — input token at step `t`, represented as a vector of dimension `d`
- `h_t ∈ ℝʰ` — hidden state at step `t`, dimension `h` (the hidden size)
- `h_{t-1}` — previous hidden state (carries memory forward)
- `W_h ∈ ℝʰˣʰ` — weight matrix applied to the previous hidden state
- `W_x ∈ ℝʰˣᵈ` — weight matrix applied to the current input
- `b ∈ ℝʰ` — bias vector
- `tanh` — squashes values to `(-1, 1)`, introduces non-linearity

The encoder produces `n` hidden states: `h₁, h₂, ..., hₙ`.

### The Bottleneck

Without attention, the decoder receives only the final hidden state `hₙ` as its starting context:

```
context = hₙ   ← single vector, fixed size h
```

**The problem:** `hₙ` must encode everything about the input sequence — all `n` tokens — into a vector of fixed dimension `h`. For long sequences, this is a lossy compression. Information from early tokens (small `t`) has been overwritten by later ones.

**Quantifying the loss:** In a vanilla RNN, the gradient of the loss with respect to `h₁` involves multiplying `W_h` by itself `(n-1)` times. If the largest singular value of `W_h` is less than 1, this product shrinks exponentially. The encoder's early states contribute almost nothing to `hₙ` for large `n`.

---

## 2. Bahdanau Attention (Additive Attention)

Bahdanau et al. (2014) proposed keeping all encoder hidden states and letting the decoder attend to them selectively.

### The Mechanism

At each decoder step `t`, instead of using only `hₙ`, compute a context vector `cₜ` as a weighted sum of all encoder states:

```
cₜ = Σᵢ αₜᵢ · hᵢ
```

**Variables:**
- `cₜ ∈ ℝʰ` — context vector for decoder step `t`
- `αₜᵢ` — attention weight: how much decoder step `t` attends to encoder position `i`
- `hᵢ ∈ ℝʰ` — encoder hidden state at position `i`
- The sum runs over all `n` encoder positions

### Computing Attention Weights

The weights `αₜᵢ` come from a softmax over alignment scores `eₜᵢ`:

```
αₜᵢ = softmax(eₜᵢ) = exp(eₜᵢ) / Σⱼ exp(eₜⱼ)
```

The alignment score `eₜᵢ` measures how well the decoder's current state `sₜ₋₁` matches encoder state `hᵢ`. Bahdanau used a small feed-forward network (the "alignment model"):

```
eₜᵢ = vₐᵀ · tanh(Wₐ · sₜ₋₁ + Uₐ · hᵢ)
```

**Variables:**
- `sₜ₋₁ ∈ ℝʰ` — decoder hidden state at the previous step
- `hᵢ ∈ ℝʰ` — encoder hidden state at position `i`
- `Wₐ ∈ ℝᵏˣʰ` — weight matrix applied to decoder state (learned)
- `Uₐ ∈ ℝᵏˣʰ` — weight matrix applied to encoder state (learned)
- `vₐ ∈ ℝᵏ` — weight vector that collapses the combined representation to a scalar (learned)
- `k` — alignment model hidden dimension (hyperparameter, typically same as `h`)

**Step by step:**
1. Project decoder state: `Wₐ · sₜ₋₁` → shape `(k,)`
2. Project encoder state: `Uₐ · hᵢ` → shape `(k,)`
3. Add them and apply tanh → shape `(k,)`, values in `(-1, 1)`
4. Dot with `vₐ` → scalar score `eₜᵢ`
5. Repeat for all `i`, collect scores `[eₜ₁, eₜ₂, ..., eₜₙ]`
6. Apply softmax → weights `[αₜ₁, αₜ₂, ..., αₜₙ]`, sum to 1
7. Weighted sum of encoder states → context vector `cₜ`

**Why tanh here?** The alignment model needs to capture non-linear interactions between the decoder state and each encoder state. A linear combination would miss cases where the relationship is more complex than "similar vectors score high."

**Tuning guide:**
- Larger `k` → more expressive alignment model, more parameters
- `k = h` is the standard default
- The alignment model adds `2kh + k` parameters — small relative to the main network

---

## 3. Dot-Product Attention (Luong Attention)

Luong and team (2015) proposed a simpler scoring function: just take the dot product between the decoder state and each encoder state.

```
eₜᵢ = sₜ · hᵢ
```

**Variables:**
- `sₜ ∈ ℝʰ` — decoder hidden state at step `t` (note: current step, not previous)
- `hᵢ ∈ ℝʰ` — encoder hidden state at position `i`
- `eₜᵢ` — scalar score (no learned parameters in the scoring function itself)

**Why this works:** The dot product measures the cosine similarity (scaled by magnitude) between two vectors. If the decoder state and an encoder state are pointing in similar directions in the hidden space, their dot product is large, meaning they're "compatible." The network learns to make compatible states point in similar directions during training.

**Worked numerical example:**

Suppose `h = 3` (hidden dimension 3 for simplicity).

Decoder state: `sₜ = [0.5, -0.2, 0.8]`

Encoder states:
```
h₁ = [0.6, -0.1, 0.7]   (similar to sₜ)
h₂ = [-0.3, 0.9, 0.1]   (dissimilar)
h₃ = [0.4, -0.3, 0.9]   (very similar)
```

Scores:
```
e₁ = 0.5×0.6 + (-0.2)×(-0.1) + 0.8×0.7 = 0.30 + 0.02 + 0.56 = 0.88
e₂ = 0.5×(-0.3) + (-0.2)×0.9 + 0.8×0.1 = -0.15 - 0.18 + 0.08 = -0.25
e₃ = 0.5×0.4 + (-0.2)×(-0.3) + 0.8×0.9 = 0.20 + 0.06 + 0.72 = 0.98
```

Softmax:
```
exp(0.88) = 2.411
exp(-0.25) = 0.779
exp(0.98) = 2.664

sum = 5.854

α₁ = 2.411 / 5.854 = 0.412
α₂ = 0.779 / 5.854 = 0.133
α₃ = 2.664 / 5.854 = 0.455
```

Context vector:
```
c = 0.412 × [0.6, -0.1, 0.7]
  + 0.133 × [-0.3, 0.9, 0.1]
  + 0.455 × [0.4, -0.3, 0.9]

c = [0.247, -0.041, 0.288]
  + [-0.040, 0.120, 0.013]
  + [0.182, -0.137, 0.410]

c = [0.389, -0.058, 0.711]
```

The context vector is dominated by `h₃` (weight 0.455) and `h₁` (weight 0.412), with `h₂` contributing little. The decoder's current state was most compatible with positions 1 and 3.

---

## 4. Scaled Dot-Product Attention

The dot product has a problem at high dimensions: scores grow large, pushing softmax into saturation.

### Why Scaling Matters

For vectors of dimension `d`, if the components are independent with mean 0 and variance 1, the dot product has variance `d`:

```
Var(q · k) = Σᵢ Var(qᵢ · kᵢ) = Σᵢ Var(qᵢ) · Var(kᵢ) = d × 1 × 1 = d
```

So the standard deviation of the dot product is `√d`. For `d = 512`, scores can easily reach magnitudes of 20–30. Softmax of `[25, 1, -3]` is approximately `[1.0, 0.0, 0.0]` — effectively a hard selection with near-zero gradients everywhere except the maximum.

**The fix:** divide by `√d` before softmax:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

After scaling, the dot products have standard deviation 1 regardless of `d`. Softmax operates in a stable range. Gradients flow.

**Worked example of the saturation problem:**

Without scaling (`d = 4`):
```
scores = [8.0, 0.5, -2.0]
softmax → [0.9994, 0.0006, 0.0000]
gradient of softmax ≈ 0 everywhere except position 0
```

With scaling (divide by `√4 = 2`):
```
scores = [4.0, 0.25, -1.0]
softmax → [0.971, 0.023, 0.007]
gradients are small but non-zero — learning can happen
```

---

## 5. The Query, Key, Value Framework

The Q/K/V abstraction generalizes attention beyond encoder-decoder pairs.

### The Full Formula

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V
```

**Variables:**
- `Q ∈ ℝⁿˣᵈₖ` — Query matrix: `n` queries, each of dimension `dₖ`
- `K ∈ ℝᵐˣᵈₖ` — Key matrix: `m` keys, each of dimension `dₖ`
- `V ∈ ℝᵐˣᵈᵥ` — Value matrix: `m` values, each of dimension `dᵥ`
- `dₖ` — key/query dimension (must match for dot product)
- `dᵥ` — value dimension (can differ from `dₖ`)
- `n` — number of queries (e.g., decoder sequence length)
- `m` — number of key-value pairs (e.g., encoder sequence length)
- Output shape: `ℝⁿˣᵈᵥ` — one output vector per query

### Step-by-Step Computation

**Step 1: Compute raw scores**
```
scores = Q · Kᵀ          shape: (n, m)
```
Each entry `scores[i, j]` is the dot product between query `i` and key `j`.

**Step 2: Scale**
```
scores = scores / √dₖ     shape: (n, m)
```

**Step 3: Apply softmax row-wise**
```
weights = softmax(scores, dim=-1)    shape: (n, m)
```
Each row sums to 1. Entry `weights[i, j]` is how much query `i` attends to position `j`.

**Step 4: Weighted sum of values**
```
output = weights · V      shape: (n, dᵥ)
```
Each output row is a weighted blend of all value vectors.

### Where Q, K, V Come From

In encoder-decoder attention:
- `Q` = linear projection of decoder states: `Q = S · Wᵠ`
- `K` = linear projection of encoder states: `K = H · Wᴷ`
- `V` = linear projection of encoder states: `V = H · Wᵛ`

Where:
- `S ∈ ℝⁿˣʰ` — decoder hidden states
- `H ∈ ℝᵐˣʰ` — encoder hidden states
- `Wᵠ ∈ ℝʰˣᵈₖ`, `Wᴷ ∈ ℝʰˣᵈₖ`, `Wᵛ ∈ ℝʰˣᵈᵥ` — learned projection matrices

**Why separate projections?** The raw hidden states encode everything. The projections let the network learn *what aspect* of each state to use as a query, key, or value. The query projection asks "what am I looking for?" The key projection asks "what do I offer?" The value projection asks "what do I actually give when selected?" These can be different subspaces of the same hidden state.

**Parameter count for one attention layer:**
```
Wᵠ: h × dₖ
Wᴷ: h × dₖ
Wᵛ: h × dᵥ
Total: h × (2dₖ + dᵥ)
```

With `h = dₖ = dᵥ = 512`: `512 × (512 + 512 + 512) = 786,432` parameters per attention layer.

---

## 6. Self-Attention

Self-attention applies the Q/K/V mechanism within a single sequence — every position attends to every other position in the same sequence.

### The Setup

Given a sequence of token representations `X ∈ ℝⁿˣᵈ` (n tokens, each of dimension d):

```
Q = X · Wᵠ     shape: (n, dₖ)
K = X · Wᴷ     shape: (n, dₖ)
V = X · Wᵛ     shape: (n, dᵥ)
```

All three matrices are derived from the same input `X`. Then:

```
SelfAttention(X) = softmax(Q·Kᵀ / √dₖ) · V    shape: (n, dᵥ)
```

**What this computes:** For each position `i`, the output is a weighted blend of all value vectors, where the weights are determined by how compatible position `i`'s query is with every other position's key. Position `i` can directly "look at" position `j` regardless of how far apart they are.

### The Attention Matrix

The weight matrix `A = softmax(Q·Kᵀ / √dₖ)` has shape `(n, n)`. Entry `A[i, j]` is how much position `i` attends to position `j`.

For a sentence of length `n`, this is an `n×n` matrix — every pair of positions has an explicit attention weight. This is what makes self-attention powerful (any-to-any connections) and expensive (O(n²) memory and computation).

**Worked example (n=4, dₖ=2):**

Input tokens (already embedded):
```
X = [[1.0, 0.0],    ← token 1
     [0.5, 0.5],    ← token 2
     [0.0, 1.0],    ← token 3
     [0.8, 0.2]]    ← token 4
```

For simplicity, assume `Wᵠ = Wᴷ = Wᵛ = I` (identity — no projection):

```
Q = K = V = X
```

Raw scores `Q·Kᵀ`:
```
scores[1,1] = [1.0,0.0]·[1.0,0.0] = 1.0
scores[1,2] = [1.0,0.0]·[0.5,0.5] = 0.5
scores[1,3] = [1.0,0.0]·[0.0,1.0] = 0.0
scores[1,4] = [1.0,0.0]·[0.8,0.2] = 0.8
```

Scale by `√2 ≈ 1.414`:
```
scaled[1,:] = [0.707, 0.354, 0.000, 0.566]
```

Softmax:
```
exp values: [2.028, 1.424, 1.000, 1.761]
sum = 6.213
weights[1,:] = [0.326, 0.229, 0.161, 0.283]
```

Output for token 1:
```
out[1] = 0.326×[1.0,0.0] + 0.229×[0.5,0.5] + 0.161×[0.0,1.0] + 0.283×[0.8,0.2]
       = [0.326, 0.000] + [0.115, 0.115] + [0.000, 0.161] + [0.226, 0.057]
       = [0.668, 0.332]
```

Token 1's representation has been updated to include information from all other tokens, weighted by compatibility.

### Computational Complexity

| Operation | Complexity |
|-----------|-----------|
| Computing Q, K, V | O(n · d · dₖ) |
| Computing scores Q·Kᵀ | O(n² · dₖ) |
| Softmax | O(n²) |
| Weighted sum | O(n² · dᵥ) |
| **Total** | **O(n² · d)** |

The O(n²) scaling is the main limitation of self-attention for very long sequences. For `n = 1000`, the attention matrix has 1 million entries. For `n = 10000`, 100 million. This is why efficient attention variants (Sparse Attention, Longformer, FlashAttention) are an active research area.

---

## 7. Multi-Head Attention

Instead of one attention operation, run `h` attention operations in parallel, each with different learned projections.

### The Formula

```
head_i = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢᵛ)

MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · Wᴼ
```

**Variables:**
- `h` — number of heads (hyperparameter; 8 in the original Transformer)
- `Wᵢᵠ ∈ ℝᵈˣᵈₖ` — query projection for head `i`
- `Wᵢᴷ ∈ ℝᵈˣᵈₖ` — key projection for head `i`
- `Wᵢᵛ ∈ ℝᵈˣᵈᵥ` — value projection for head `i`
- `Wᴼ ∈ ℝ⁽ʰᵈᵥ⁾ˣᵈ` — output projection that combines all heads
- `dₖ = dᵥ = d/h` — standard choice: split the model dimension evenly across heads

### Dimension Arithmetic

With model dimension `d = 512` and `h = 8` heads:
```
dₖ = dᵥ = 512 / 8 = 64

Each head:
  Wᵢᵠ: 512 × 64
  Wᵢᴷ: 512 × 64
  Wᵢᵛ: 512 × 64
  head_i output: (n, 64)

After concat: (n, 8×64) = (n, 512)
Wᴼ: 512 × 512

Total parameters per multi-head attention layer:
  8 heads × 3 projections × (512×64) + output projection (512×512)
= 8 × 3 × 32,768 + 262,144
= 786,432 + 262,144
= 1,048,576  ≈ 1M parameters
```

**Key insight:** Using `h` heads with dimension `d/h` each costs the same as one head with dimension `d`. The total parameter count is the same. But you get `h` different learned subspaces — each head can specialize in a different type of relationship.

### Why Multiple Heads?

A single attention head computes one weighted average over the values. It can only "look for" one type of relationship at a time.

Multiple heads allow the model to simultaneously attend to:
- Syntactic relationships (subject-verb agreement)
- Semantic relationships (word meaning similarity)
- Positional relationships (nearby words)
- Coreference (pronoun → noun)

Each head learns its own `Wᵠ`, `Wᴷ`, `Wᵛ` projections, so it learns to ask different questions and find different patterns. The output projection `Wᴼ` then combines all these perspectives into a single representation.

**Analogy to CNNs:** A single conv filter detects one pattern. Multiple filters detect multiple patterns in parallel. Multi-head attention is the same idea applied to sequence relationships instead of spatial patterns.

### Parameter Count Summary

For a multi-head attention layer with model dimension `d`, `h` heads, `dₖ = dᵥ = d/h`:

```
Query projections:  h × (d × dₖ) = h × d × (d/h) = d²
Key projections:    h × (d × dₖ) = d²
Value projections:  h × (d × dᵥ) = d²
Output projection:  (h × dᵥ) × d = d²

Total: 4d²
```

For `d = 512`: `4 × 512² = 1,048,576` ≈ 1M parameters per attention layer.

---

## 8. The Attention Weight Matrix: What It Tells You

The attention weight matrix `A ∈ ℝⁿˣⁿ` (for self-attention) or `A ∈ ℝⁿˣᵐ` (for cross-attention) is interpretable.

### Reading the Matrix

- **Row `i`**: the attention distribution for position `i` — where it looks
- **Column `j`**: how much all positions attend to position `j` — how "attended to" it is
- **Diagonal entries**: how much each position attends to itself
- **Off-diagonal entries**: cross-position attention

### Patterns to Look For

| Pattern | What it means |
|---------|--------------|
| Strong diagonal | Each position mostly attends to itself — little cross-position interaction |
| Off-diagonal blocks | Groups of positions attending to each other — phrase-level structure |
| Specific off-diagonal entries | Direct relationships (pronoun → noun, verb → subject) |
| Uniform rows | The head hasn't learned to be selective — may be underutilized |
| Near-zero rows | Numerical instability — check scaling |

### For Encoder-Decoder Attention (Translation)

The weight matrix `A ∈ ℝⁿˣᵐ` (output length × input length) shows the alignment between source and target.

- **Diagonal pattern**: source and target have similar word order (e.g., English→French)
- **Non-diagonal pattern**: word order differs (e.g., Tamil→English, SOV→SVO)
- **Monotone pattern**: strict left-to-right alignment
- **Scattered pattern**: long-range dependencies, reordering

---

## 9. Masking in Attention

Two types of masking are used in practice.

### Padding Mask

Sequences in a batch have different lengths. Shorter sequences are padded with a special token. We don't want the model to attend to padding positions.

**Implementation:** Set the score for padding positions to `-∞` before softmax:

```
scores[i, j] = -∞   if position j is a padding token
```

After softmax, `exp(-∞) = 0`, so padding positions receive zero attention weight.

### Causal (Look-Ahead) Mask

In language generation, the model should only attend to past positions — it can't see future tokens it hasn't generated yet.

**Implementation:** Set the upper triangle of the score matrix to `-∞`:

```
scores[i, j] = -∞   if j > i   (future position)
```

This creates a lower-triangular attention pattern — each position can only attend to itself and earlier positions.

```
Causal attention matrix (n=4):
Position:  1    2    3    4
1:       [0.8  -∞   -∞   -∞ ]   → attends only to position 1
2:       [0.3  0.7  -∞   -∞ ]   → attends to positions 1-2
3:       [0.1  0.4  0.5  -∞ ]   → attends to positions 1-3
4:       [0.2  0.3  0.2  0.3]   → attends to all positions
```

---

## 10. Gradient Flow Through Attention

One reason attention works well: gradients flow directly from the output back to any input position.

### The Gradient Path

In a standard RNN, the gradient from the loss to encoder position `i` must travel through `(n - i)` recurrent steps — each multiplying by `W_h`. For large `(n - i)`, this vanishes.

In attention, the gradient from the output at decoder step `t` to encoder state `hᵢ` has two paths:

```
cₜ = Σⱼ αₜⱼ · hⱼ

∂cₜ/∂hᵢ = αₜᵢ · I  +  Σⱼ (∂αₜⱼ/∂hᵢ) · hⱼᵀ
             ↑                    ↑
        direct path         indirect path (αₜⱼ depends on hᵢ through scores)
```

The first term is the direct contribution: `hᵢ` appears in the weighted sum with weight `αₜᵢ`. The second term accounts for the fact that changing `hᵢ` also changes the attention weights themselves (since `hᵢ` participates in the score computation via the key).

In practice, the direct term `αₜᵢ · I` dominates and is the key insight. The gradient flows directly from the output to any input position, scaled by the attention weight. No chain of matrix multiplications. No vanishing. If `αₜᵢ > 0` (which it always is after softmax), the gradient is non-zero.

**This is the fundamental reason attention solves the long-range dependency problem** — not just for the forward pass (context vector), but for the backward pass (gradient flow) as well.

---

## 11. Complexity Comparison

| Model | Sequential ops | Memory (per step) | Long-range dependency |
|-------|---------------|--------|----------------------|
| RNN | O(n) | O(h) state; O(n·h) for backprop | O(n) path length |
| LSTM | O(n) | O(h) state; O(n·h) for backprop | O(n) path length |
| Attention | O(1) | O(n²) | O(1) path length |

**Reading the table:**
- RNN/LSTM: sequential — can't parallelize across time steps; long-range dependencies require gradients to travel O(n) steps. Memory per step is O(h) during inference, but training requires storing all n hidden states for backpropagation, giving O(n·h) total
- Attention: fully parallel — all positions computed simultaneously; any two positions are directly connected (O(1) path length), but requires O(n²) memory for the attention matrix

The O(n²) memory cost is why attention is expensive for very long sequences (documents, audio, genomics). The O(1) path length is why it handles long-range dependencies so much better than RNNs.

---

## 12. What Attention Still Needs

Attention alone is not a complete architecture. Two things are missing:

### Position Information

The attention operation is **permutation-equivariant**: if you shuffle the input tokens, the output tokens shuffle in the same way. The attention weights don't change — the dot products between Q and K don't depend on position.

Language is not permutation-invariant. "The cat sat on the mat" ≠ "The mat sat on the cat."

**Fix:** Add positional encodings to the input before attention. The Transformer uses sinusoidal encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where `pos` is the position and `i` is the dimension index. These are added to the token embeddings before the first attention layer. Post 10 covers this in detail.

### Non-Linearity Between Layers

The attention output is a weighted sum — a linear operation. Stacking multiple attention layers without non-linearity between them collapses to a single linear transformation (no expressive power gain from depth).

**Fix:** Add a position-wise feed-forward network after each attention layer:

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

This is just a two-layer MLP applied independently to each position. It introduces non-linearity and allows each position to transform its representation after the attention mixing step.

Both of these — positional encodings and the feed-forward sublayer — are core components of the Transformer architecture. Post 10.

---

## 13. Illustrative Examples

This section walks through three end-to-end examples that tie together the concepts above. Each one is designed to build intuition for a different aspect of attention.

### Example A: Multi-Head Attention on a 3-Word Sentence

This example shows how two attention heads can learn to focus on different relationships in the same input.

**Setup:** 3 tokens ("The", "cat", "sat"), `h = 2` heads, `dₖ = dᵥ = 2` per head. We show the already-projected Q, K, V for each head (i.e., after multiplying by the learned `Wᵢᵠ`, `Wᵢᴷ`, `Wᵢᵛ` matrices).

**Head 1** (learns syntactic subject-verb relationships):
```
Q₁ = [[ 0.1,  1.5],    ← "The"     K₁ = [[-0.2,  1.2],    V₁ = [[0.3, 0.7],
      [ 1.5,  0.2],    ← "cat"           [ 1.6,  0.1],          [0.9, 0.1],
      [ 1.8, -0.3]]    ← "sat"           [ 0.3,  0.8]]          [0.5, 0.5]]
```

**Head 1 scores** (Q₁ · K₁ᵀ, then scale by √2):
```
                    "The"   "cat"   "sat"
"The" scores:  [  1.780,  0.310,  1.230]  → weights: [0.492, 0.174, 0.334]
"cat" scores:  [ -0.060,  2.420,  0.610]  → weights: [0.119, 0.689, 0.192]
"sat" scores:  [ -0.720,  2.850,  0.300]  → weights: [0.064, 0.803, 0.132]
                                                              ↑
                                              "sat" attends strongly to "cat" (subject-verb)
```

Head 1 output for "sat":
```
out₁[sat] = 0.064×[0.3,0.7] + 0.803×[0.9,0.1] + 0.132×[0.5,0.5]
          = [0.019, 0.045] + [0.723, 0.080] + [0.066, 0.066]
          = [0.808, 0.191]
```

**Head 2** (learns positional proximity):
```
Q₂ = [[ 1.0,  0.5],    ← "The"     K₂ = [[ 0.8,  0.6],    V₂ = [[0.8, 0.2],
      [ 0.5,  1.0],    ← "cat"           [ 0.6,  0.8],          [0.4, 0.6],
      [-0.5,  1.5]]    ← "sat"           [-0.3,  1.4]]          [0.1, 0.9]]
```

**Head 2 scores:**
```
                    "The"   "cat"   "sat"
"The" scores:  [  1.100,  1.000,  0.400]  → weights: [0.393, 0.367, 0.240]
"cat" scores:  [  1.000,  1.100,  1.250]  → weights: [0.306, 0.329, 0.365]
"sat" scores:  [  0.500,  0.900,  2.250]  → weights: [0.173, 0.230, 0.597]
                                                                      ↑
                                              "sat" attends most to itself (local context)
```

Head 2 output for "sat":
```
out₂[sat] = 0.173×[0.8,0.2] + 0.230×[0.4,0.6] + 0.597×[0.1,0.9]
          = [0.138, 0.035] + [0.092, 0.138] + [0.060, 0.537]
          = [0.290, 0.710]
```

**Concatenation for "sat":**
```
Concat(head₁, head₂) = [0.808, 0.191, 0.290, 0.710]  (back to d=4)
```

Then `Wᴼ ∈ ℝ⁴ˣ⁴` maps this back to the model dimension. The key takeaway: Head 1 learned that "sat" should look at "cat" (the subject of the verb), while Head 2 learned that "sat" should focus on local context. These are two different types of relationships captured simultaneously. In a real Transformer with 8+ heads and larger dimensions, the specialization is even more dramatic — some heads track syntax, others track coreference, others track positional proximity.

### Example B: Causal Masking Step by Step

This example shows exactly how the causal mask prevents information leakage during autoregressive generation.

**Setup:** Generating a 4-token sequence. `dₖ = 2`. We're at the stage where all 4 tokens exist and we want to compute masked self-attention.

```
Q = [[0.9, 0.1],    ← position 1
     [0.3, 0.7],    ← position 2
     [0.5, 0.5],    ← position 3
     [0.1, 0.9]]    ← position 4

K = Q  (self-attention, identity projection for simplicity)
V = Q
```

**Step 1: Raw scores** `Q · Kᵀ`:
```
        pos1   pos2   pos3   pos4
pos1: [ 0.82   0.34   0.50   0.18 ]
pos2: [ 0.34   0.58   0.50   0.66 ]
pos3: [ 0.50   0.50   0.50   0.50 ]
pos4: [ 0.18   0.66   0.50   0.82 ]
```

**Step 2: Scale** by `√2 ≈ 1.414`:
```
        pos1   pos2   pos3   pos4
pos1: [ 0.580  0.240  0.354  0.127 ]
pos2: [ 0.240  0.410  0.354  0.467 ]
pos3: [ 0.354  0.354  0.354  0.354 ]
pos4: [ 0.127  0.467  0.354  0.580 ]
```

**Step 3: Apply causal mask** (set future positions to -∞):
```
        pos1   pos2   pos3   pos4
pos1: [ 0.580   -∞     -∞     -∞  ]
pos2: [ 0.240  0.410   -∞     -∞  ]
pos3: [ 0.354  0.354  0.354   -∞  ]
pos4: [ 0.127  0.467  0.354  0.580]
```

**Step 4: Softmax** (row-wise, -∞ → 0 weight):
```
        pos1   pos2   pos3   pos4
pos1: [ 1.000  0.000  0.000  0.000 ]   ← can only see itself
pos2: [ 0.458  0.542  0.000  0.000 ]   ← sees pos 1-2
pos3: [ 0.333  0.333  0.333  0.000 ]   ← sees pos 1-3 (equal scores!)
pos4: [ 0.191  0.268  0.240  0.301 ]   ← sees all positions
```

**Step 5: Weighted sum** with V:
```
out[1] = 1.000×[0.9,0.1] = [0.900, 0.100]
out[2] = 0.458×[0.9,0.1] + 0.542×[0.3,0.7] = [0.575, 0.425]
out[3] = 0.333×[0.9,0.1] + 0.333×[0.3,0.7] + 0.333×[0.5,0.5] = [0.567, 0.433]
out[4] = 0.191×[0.9,0.1] + 0.268×[0.3,0.7] + 0.240×[0.5,0.5] + 0.301×[0.1,0.9]
       = [0.172,0.019] + [0.080,0.188] + [0.120,0.120] + [0.030,0.271]
       = [0.403, 0.597]
```

Notice how position 1's output is exactly its own value — it has no context from other tokens. Position 4 has the richest representation because it can attend to the entire sequence. This asymmetry is fundamental to autoregressive models: later positions always have more context than earlier ones.

### Example C: Why Scaling Matters — A Side-by-Side Comparison

This example uses the same Q and K vectors but compares attention at `dₖ = 2` vs `dₖ = 64` to show why `√dₖ` scaling is essential.

**Low dimension (dₖ = 2):**
```
q = [0.8, 0.6]
k₁ = [0.9, 0.5]    k₂ = [0.1, 0.7]    k₃ = [-0.3, 0.4]

scores:
  q·k₁ = 0.72 + 0.30 = 1.02
  q·k₂ = 0.08 + 0.42 = 0.50
  q·k₃ = -0.24 + 0.24 = 0.00

Without scaling → softmax([1.02, 0.50, 0.00]) = [0.511, 0.304, 0.184]
With scaling /√2 → softmax([0.72, 0.35, 0.00]) = [0.459, 0.318, 0.223]
```

Difference is modest. Both distributions are reasonably spread out.

**High dimension (dₖ = 64):**

Now imagine q and k are 64-dimensional with the same component statistics (mean 0, variance 1). The dot product variance is 64, so typical scores are around `±√64 = ±8`.

```
Typical scores: [18.3, 2.1, -5.7]

Without scaling → softmax([18.3, 2.1, -5.7]) = [1.0000, 0.0000, 0.0000]
  (exp(18.3) ≈ 88 million — completely dominates)

With scaling /√64 → softmax([2.29, 0.26, -0.71]) = [0.846, 0.112, 0.042]
  (gradients flow to all three positions)
```

**The gradient problem in numbers:**

The softmax gradient for position `j` is `αⱼ(1 - αⱼ)` (diagonal term of the Jacobian). When `αⱼ ≈ 1.0`, the gradient is `1.0 × 0.0 = 0.0`. When `αⱼ ≈ 0.0`, the gradient is also `0.0 × 1.0 = 0.0`. Only when `αⱼ` is in a moderate range (say 0.1 to 0.9) do gradients flow meaningfully.

```
Without scaling (d=64):  α = [1.000, 0.000, 0.000]
  gradients: [0.000, 0.000, 0.000]  ← dead, no learning

With scaling (d=64):     α = [0.846, 0.112, 0.042]
  gradients: [0.130, 0.099, 0.040]  ← alive, learning happens
```

This is why the `1/√dₖ` factor isn't optional — it's load-bearing. Without it, attention in high dimensions collapses to hard argmax selection and gradients vanish.

### Example D: Cross-Attention in Translation (Tamil → English)

This example traces a single decoder step during translation, showing how the decoder query selects relevant encoder positions.

**Setup:** Translating "நாளைக்குள் report அனுப்பு" (Send the report by tomorrow) to English. The encoder has produced hidden states for 3 Tamil tokens. The decoder is generating the English word "Send".

Encoder hidden states (after projection to keys and values), `dₖ = 3`:
```
K = [[0.8, -0.1, 0.3],    ← "நாளைக்குள்" (by tomorrow) — temporal marker
     [0.1, 0.7, 0.5],     ← "report" — object noun
     [0.6, 0.2, 0.9]]     ← "அனுப்பு" (send) — verb

V = [[0.2, 0.9, 0.1],     ← value for "நாளைக்குள்"
     [0.8, 0.1, 0.6],     ← value for "report"
     [0.3, 0.5, 0.8]]     ← value for "அனுப்பு"
```

Decoder query for generating "Send", `dₖ = 3`:
```
q = [0.7, 0.1, 0.8]   ← the decoder is "asking for" a verb-like representation
```

**Score computation:**
```
q · k₁ = 0.7×0.8 + 0.1×(-0.1) + 0.8×0.3 = 0.56 - 0.01 + 0.24 = 0.79
q · k₂ = 0.7×0.1 + 0.1×0.7 + 0.8×0.5 = 0.07 + 0.07 + 0.40 = 0.54
q · k₃ = 0.7×0.6 + 0.1×0.2 + 0.8×0.9 = 0.42 + 0.02 + 0.72 = 1.16
```

**Scale by √3 ≈ 1.732:**
```
scaled = [0.79/1.732, 0.54/1.732, 1.16/1.732] = [0.456, 0.312, 0.670]
```

**Softmax:**
```
exp values: [1.578, 1.366, 1.954]
sum = 4.898
α = [0.322, 0.279, 0.399]
```

**Context vector:**
```
c = 0.322×[0.2, 0.9, 0.1] + 0.279×[0.8, 0.1, 0.6] + 0.399×[0.3, 0.5, 0.8]
  = [0.064, 0.290, 0.032] + [0.223, 0.028, 0.167] + [0.120, 0.199, 0.319]
  = [0.407, 0.517, 0.519]
```

The decoder attends most strongly to "அனுப்பு" (send, weight 0.399) — exactly the Tamil word that corresponds to the English word "Send" being generated. The attention has learned the cross-lingual alignment: when generating a verb in English, look at the verb in Tamil, even though it appears at a different position in the sentence.

---