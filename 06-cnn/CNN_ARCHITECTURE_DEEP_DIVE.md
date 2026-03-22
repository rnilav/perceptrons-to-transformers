# CNN Architecture Deep Dive

A companion to [Blog Post #6: Convolutional Neural Networks](blog-post-cnn.md).

This document covers the full mathematical treatment of convolution, pooling, parameter counting, and a detailed comparison of major CNN architectures.

---

## 1. Convolution: The Full Math

### The Operation

Given:
- Input of shape `(H_in, W_in, C_in)` — height, width, number of channels (e.g. 3 for RGB)
- A filter (kernel) of shape `(K, K, C_in)` — a small grid of learnable numbers, same depth as input
- Stride `S` — how many pixels the filter moves at each step
- Padding `P` — how many zeros to add around the border of the input

The output at position `(i, j)` for filter `f` is:

```
output[i, j, f] = Σ_c Σ_m Σ_n  input[i*S + m, j*S + n, c] * filter[m, n, c, f]  +  bias[f]
```

Reading this in plain English: for each output position `(i, j)`, slide the filter to that location (scaled by stride `S`), multiply every filter weight by the corresponding input value, sum everything up across all channels `c` and all filter positions `(m, n)`, then add a bias. The result is a single number — how strongly this filter's pattern was detected at this location.

Where `m, n` range over the kernel dimensions `[0, K)`.

### Output Dimensions

```
H_out = floor((H_in - K + 2*P) / S) + 1
W_out = floor((W_in - K + 2*P) / S) + 1
```

**What each term does:**

| Term | Role | Effect |
|------|------|--------|
| `H_in` | input height | starting size |
| `- K` | kernel size | filter can't be centered on the last K/2 pixels without padding |
| `+ 2*P` | padding on both sides | adds P zeros on left and P on right, recovering lost border |
| `/ S` | stride | larger stride = fewer positions the filter can land on |
| `+ 1` | count correction | the first position counts too (off-by-one fix) |

**Intuition:** The formula counts how many times you can place the filter, starting from position 0, stepping by S, before you fall off the edge. Padding extends the edge so you don't lose information. Stride controls how densely you sample.

**Tuning guide:**
- Want output same size as input? Use `P = (K-1)/2` with `S=1` (called "same" padding)
- Want to halve spatial dimensions? Use `S=2` (common after conv blocks)
- Want to preserve detail? Use small stride (S=1) and add padding
- Want to aggressively downsample? Use large stride or no padding

### Worked Example

Input: 5×5, Kernel: 3×3, Padding: 0, Stride: 1

```
H_out = (5 - 3 + 0) / 1 + 1 = 3
W_out = (5 - 3 + 0) / 1 + 1 = 3  →  output is 3×3
```

Input: 5×5, Kernel: 3×3, Padding: 1, Stride: 1

```
H_out = (5 - 3 + 2) / 1 + 1 = 5
W_out = (5 - 3 + 2) / 1 + 1 = 5  →  output preserves spatial size
```

Input: 28×28, Kernel: 5×5, Padding: 0, Stride: 1

```
H_out = (28 - 5 + 0) / 1 + 1 = 24  →  output is 24×24
```

### Parameter Count for a Conv Layer

```
params = (K * K * C_in + 1) * C_out
         └─────────────────┘   └────┘
           weights per filter   bias per filter
```

**Why this formula:** Each filter is a `K×K` grid applied to all `C_in` input channels — so `K*K*C_in` weights. Each filter also has one bias term (`+1`). Multiply by `C_out` because we learn that many independent filters. The key insight: this count doesn't depend on the input image size `H_in` or `W_in` at all. The same 896 parameters work whether the image is 28×28 or 224×224. That's weight sharing.

Example: 32 filters of 3×3 on a 3-channel (RGB) input:
```
params = (3 * 3 * 3 + 1) * 32 = 28 * 32 = 896
```

**Tuning guide:**
- More filters (`C_out`) = more patterns the layer can detect, but more parameters and computation
- Larger kernel (`K`) = larger receptive field per layer, but more parameters; 3×3 is the modern default
- Stacking two 3×3 layers gives the same receptive field as one 5×5 with fewer params and an extra non-linearity

---

## 2. Pooling: The Full Math

### Max Pooling

Given pool size `P` and stride `S`:

```
output[i, j] = max over m in [0,P), n in [0,P) of  input[i*S + m, j*S + n]
```

**Plain English:** Place a `P×P` window at position `(i*S, j*S)` in the input. Take the single largest value inside that window. That's your output at `(i, j)`. No weights, no learning — just selection.

**Why max and not average?** Max pooling asks "was this feature present anywhere in this region?" — it preserves the strongest signal. Average pooling asks "how active was this region on average?" — it smooths. Max pooling is preferred for feature detection because a strong activation in one corner of the window is more informative than a weak average across the whole window.

### Output Dimensions

```
H_out = floor((H_in - P) / S) + 1
W_out = floor((W_in - P) / S) + 1
```

**What each term does:**

| Term | Role |
|------|------|
| `H_in` | input height |
| `- P` | window can't start at the last P-1 rows (would fall off the edge) |
| `/ S` | stride controls how many non-overlapping (or overlapping) windows fit |
| `+ 1` | count the first window |

**Tuning guide:**
- `P=2, S=2` is the standard: halves spatial dimensions, no overlap between windows
- `P=3, S=2` is used in AlexNet: slight overlap between windows, slightly more information retained
- `P=2, S=1` gives overlapping windows: output barely shrinks, rarely used
- Larger `P` = more aggressive downsampling = more translation invariance, but more information loss

### Worked Example

Input: 4×4, Pool: 2×2, Stride: 2

```
H_out = (4 - 2) / 2 + 1 = 2
W_out = (4 - 2) / 2 + 1 = 2  →  output is 2×2
```

```
Input:              Max Pool (2×2, stride=2):
[1  3  2  4]        [6  4]
[5  6  1  2]   →    [8  7]
[3  8  4  7]
[1  2  6  3]
```

Top-left window `[1,3,5,6]` → max is 6. Top-right `[2,4,1,2]` → max is 4. And so on.

Pooling has **zero learnable parameters**. It's a fixed operation — nothing to train, nothing to overfit.

---

## 3. FC Network vs CNN: Parameter Comparison

The core difference: an FC layer connects every input to every output with a unique weight. A conv layer connects each output to a small local patch of the input, and reuses the same weights everywhere. That reuse is why CNNs are so much more efficient.

### Fully-Connected Network on 224×224×3 Image

```
Layer               Shape           Parameters
─────────────────────────────────────────────────
Input               150,528         —
FC Hidden (1000)    1,000           150,528 × 1,000 + 1,000 = 150,529,000
FC Hidden (1000)    1,000           1,000 × 1,000 + 1,000   = 1,001,000
FC Output (1000)    1,000           1,000 × 1,000 + 1,000   = 1,001,000
─────────────────────────────────────────────────
TOTAL                               ~152.5 million
```

150M parameters just to process one image — and the first layer has to learn from scratch that nearby pixels are related. The architecture gives it no help.

### LeNet on 28×28×1 Image (1998)

```
Layer                   Output Shape    Parameters
──────────────────────────────────────────────────────────
Input                   28×28×1         —
Conv(6, 5×5)            24×24×6         (5×5×1 + 1) × 6   = 156
MaxPool(2×2)            12×12×6         0
Conv(16, 5×5)           8×8×16          (5×5×6 + 1) × 16  = 2,416
MaxPool(2×2)            4×4×16          0
Flatten                 256             —
FC(120)                 120             256 × 120 + 120    = 30,840
FC(84)                  84              120 × 84 + 84      = 10,164
FC(10)                  10              84 × 10 + 10       = 850
──────────────────────────────────────────────────────────
TOTAL                                   ~60,000
```

Notice: the two conv layers together use only 2,572 parameters. The FC layers use the rest. Even in 1998, the bottleneck was the fully-connected classifier at the end, not the convolutional feature extractor.

### AlexNet on 224×224×3 Image (2012)

```
Layer                           Output Shape    Parameters
──────────────────────────────────────────────────────────────────
Input                           224×224×3       —
Conv(96, 11×11, S=4)            54×54×96        (11×11×3+1)×96    = 34,944
MaxPool(3×3, S=2)               26×26×96        0
Conv(256, 5×5, P=2)             26×26×256       (5×5×96+1)×256    = 614,656
MaxPool(3×3, S=2)               12×12×256       0
Conv(384, 3×3, P=1)             12×12×384       (3×3×256+1)×384   = 885,120
Conv(384, 3×3, P=1)             12×12×384       (3×3×384+1)×384   = 1,327,488
Conv(256, 3×3, P=1)             12×12×256       (3×3×384+1)×256   = 884,992
MaxPool(3×3, S=2)               5×5×256         0
Flatten                         6,400           —
FC(4096)                        4,096           6,400×4,096+4,096 = 26,218,496
FC(4096)                        4,096           4,096×4,096+4,096 = 16,781,312
FC(1000)                        1,000           4,096×1,000+1,000 = 4,097,000
──────────────────────────────────────────────────────────────────
TOTAL                                           ~60 million
```

The pattern from LeNet repeats at scale: the conv layers (~3.7M params) do the feature extraction; the FC layers (~47M params) do the classification. The large 11×11 first filter with stride=4 aggressively downsamples the 224×224 input early — a design choice driven by the GPU memory constraints of 2012.

Key innovations over LeNet:
- ReLU activations (faster training, no vanishing gradient — same ReLU from Post 2)
- Dropout in FC layers (regularization — same dropout from Post 5)
- GPU training (10-50× speedup, enabled the scale)
- Data augmentation (random crops, flips — effectively multiplies dataset size)

---

## 4. VGG: Depth Through Simplicity (2014)

VGG's insight: **use only 3×3 filters, stack more layers**.

**Why 3×3?** Two 3×3 conv layers have the same receptive field as one 5×5 layer — both "see" a 5×5 region of the original input. But two 3×3 layers have fewer parameters and an extra non-linearity (ReLU between them), which makes the function more expressive.

```
Two 3×3 layers:  2 × (3×3×C×C) = 18C²  params
One 5×5 layer:   1 × (5×5×C×C) = 25C²  params
Saving: 28% fewer params, plus an extra ReLU
```

This principle generalizes: three 3×3 layers ≈ one 7×7 layer, with even greater savings. VGG standardized this as the default design pattern.

### VGG-16 Architecture (simplified)

```
Block 1: Conv(64,3×3) → Conv(64,3×3) → MaxPool       →  112×112×64
Block 2: Conv(128,3×3) → Conv(128,3×3) → MaxPool      →  56×56×128
Block 3: Conv(256,3×3) × 3 → MaxPool                  →  28×28×256
Block 4: Conv(512,3×3) × 3 → MaxPool                  →  14×14×512
Block 5: Conv(512,3×3) × 3 → MaxPool                  →  7×7×512
FC(4096) → FC(4096) → FC(1000)

Total: ~138 million parameters
Top-5 accuracy on ImageNet: 92.7%
```

The problem: those three FC layers hold ~123M of the 138M parameters. The conv layers do the real work; the FC layers are the bottleneck. VGG proved depth matters, but also exposed that large FC classifiers are expensive and don't generalize as well as the conv features themselves.

**Tuning insight:** VGG's uniform structure (same filter size everywhere, double channels after each pool) became a template. When in doubt, double the channels when you halve the spatial size — this keeps the total "information capacity" roughly constant through the network.

---

## 5. ResNet: Depth Through Residual Connections (2015)

VGG showed depth helps. But beyond ~20 layers, a strange thing happens: adding more layers makes accuracy *worse*, even on training data. This isn't overfitting — the training loss itself gets worse. It's a pure optimization problem.

**Why?** Backpropagation multiplies gradients as it flows backward through layers. With many layers, those multiplications shrink the gradient exponentially. By the time the signal reaches early layers, it's nearly zero. Early layers stop learning. The network degrades.

ResNet's solution: **skip connections** that give gradients a direct path home.

### The Residual Block

Instead of learning `H(x)` directly, learn the residual `F(x) = H(x) - x`:

```
Standard block:          Residual block:

x → [Conv→BN→ReLU]       x ──────────────────┐
  → [Conv→BN→ReLU]         → [Conv→BN→ReLU]  │
  → output                 → [Conv→BN]        │
                           → (+) ← ───────────┘
                           → ReLU
                           → output
```

**Why this works:** If the optimal transformation is close to identity (the layer shouldn't change much), the residual `F(x)` just needs to learn zero — which is easy. Without skip connections, the network has to learn the identity mapping through a stack of non-linear layers — which is surprisingly hard.

Think of it like this: it's easier to learn "what to add" than "what the output should be from scratch."

### Why This Matters for Gradients

During backpropagation, the skip connection creates a direct gradient path:

```
∂L/∂x = ∂L/∂output × (∂F(x)/∂x + 1)
                                   ↑
                         gradient always has +1
                         even if ∂F/∂x ≈ 0
```

The `+1` term means gradients can never fully vanish. Even if the learned transformation `F(x)` contributes nothing, the gradient still flows through the skip connection unchanged. Early layers always receive a meaningful signal.

### ResNet-50 Architecture

```
Input: 224×224×3
Conv(64, 7×7, S=2) → MaxPool(3×3, S=2)   →  56×56×64

Layer 1: 3 × Bottleneck(64)               →  56×56×256
Layer 2: 4 × Bottleneck(128, S=2)         →  28×28×512
Layer 3: 6 × Bottleneck(256, S=2)         →  14×14×1024
Layer 4: 3 × Bottleneck(512, S=2)         →  7×7×2048

GlobalAvgPool → FC(1000)

Total: ~25 million parameters
Top-5 accuracy on ImageNet: 96.4%
```

Notice: no large FC layers. ResNet replaces the VGG-style `FC(4096) → FC(4096)` with a single `GlobalAvgPool → FC(1000)`. Global average pooling takes the spatial average of each feature map — one number per channel — collapsing `7×7×2048` to just `2048`. This eliminates ~47M parameters while improving generalization.

A Bottleneck block uses 1×1 convolutions to reduce and restore channel dimensions, making deep networks computationally feasible:

```
Bottleneck(256 channels):
  1×1 Conv(64)   ← reduce channels (cheap: no spatial computation)
  3×3 Conv(64)   ← spatial convolution (expensive, but on fewer channels)
  1×1 Conv(256)  ← restore channels
  + skip connection

Cost: 3×3×64×64 = 36,864 multiplications
vs. plain 3×3 Conv(256): 3×3×256×256 = 589,824 multiplications  →  16× cheaper
```

**Tuning insight:** The bottleneck ratio (here 4:1, reducing 256 → 64) controls the trade-off between computation and capacity. Wider bottlenecks (smaller reduction) are more expressive but more expensive. The 4:1 ratio is the standard default.

---

## 6. Architecture Comparison Summary

```
Architecture  Year  Layers  Parameters  ImageNet Top-5  Key Innovation
──────────────────────────────────────────────────────────────────────────
LeNet         1998  8       ~60K        99% (MNIST)     First practical CNN
AlexNet       2012  8       ~60M        84.7%           ReLU, GPU, Dropout
VGG-16        2014  16      ~138M       92.7%           3×3 filters, depth
ResNet-50     2015  50      ~25M        96.4%           Residual connections
ResNet-152    2015  152     ~60M        96.9%           Extreme depth
```

The progression tells a clear story:
- **LeNet → AlexNet**: scale up, add ReLU and dropout, use GPUs
- **AlexNet → VGG**: go deeper with uniform 3×3 filters
- **VGG → ResNet**: solve the depth problem with skip connections, use fewer parameters more efficiently

ResNet-50 achieves better accuracy than VGG-16 with 5× fewer parameters. The architecture matters more than raw size.

---

## 7. Receptive Field Growth

A key property of stacked convolutions: the effective receptive field grows with depth, even though each individual filter is small.

```
Layer 1 (3×3 conv):  sees 3×3 patch of input
Layer 2 (3×3 conv):  sees 5×5 patch of input
Layer 3 (3×3 conv):  sees 7×7 patch of input
Layer k (3×3 conv):  sees (2k+1)×(2k+1) patch of input
```

**Why:** Each layer-2 neuron sees a 3×3 region of layer-1 outputs. Each layer-1 output saw a 3×3 region of the input. So layer-2 effectively sees a 5×5 region of the input (3 + 2×1 = 5, adding one pixel of context on each side per layer).

With pooling, the receptive field grows even faster. A 2×2 max pool doubles the effective receptive field of all subsequent layers — because each pooled value represents a 2×2 region of the previous feature map.

**Why this matters:** Early layers detect small local patterns (edges, corners). Later layers, with large receptive fields, can detect large patterns (whole objects, scenes) — even though every individual filter is still just 3×3. The hierarchy of features emerges naturally from the stacking, not from using large filters.

**Tuning insight:** If your network isn't detecting large-scale patterns well, you may need more layers (larger receptive field) rather than larger filters. Larger filters are expensive; more layers are cheaper and more expressive.
