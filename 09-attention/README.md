# Post 9: Attention Mechanisms

From compressed summaries to direct access. This post introduces attention — the idea that a decoder doesn't have to rely on a single fixed-size vector, but can look back at any part of the input it needs.

📖 [Blog Post](https://dev.to/rnilav/attention-mechanisms-stop-compressing-start-looking-back-1bol) 

## Setup

```bash
pip install -r requirements.txt
streamlit run attention_playground.py
```

Dependencies: `numpy`, `matplotlib`, `streamlit`. No TensorFlow or GPU required.

---

## The Playground

Five concept demos that follow the blog post narrative. No training loops, no waiting — every interaction is instant because it's all just matrix math.

### Demo 1 — The Bottleneck

A bar chart showing how much each token contributes to the RNN encoder's final hidden state. Drag the sentence length slider and watch early tokens fade to near-zero. The decay is exponential and independent of hidden state size — a 1024-dimensional vector overwrites early tokens just as aggressively as a 16-dimensional one.

### Demo 2 — Dot-Product Attention

Pick a sentence and a query word. See the scaled dot-product scores and the resulting attention weights as a colour bar. Below, a separate panel lets you slide the embedding dimension from d=4 to d=512 and watch unscaled softmax collapse to one-hot while scaled softmax stays smooth — the √d effect, live.

### Demo 3 — Word Reordering (Cross-Attention)

The Tamil→English translation example from the blog post, shown as two alignment matrices side by side. The left matrix shows a naive left-to-right alignment (wrong). The right matrix shows the learned attention pattern — the decoder jumping from position 5 back to 4, then 3, then 1 to follow English word order. The non-diagonal pattern is attention doing the reordering.

### Demo 4 — Self-Attention Heatmap

The full n×n self-attention matrix for a sentence. Pick any word to highlight its row and see a companion bar chart of where it looks. Toggle the causal mask to see the lower-triangular pattern autoregressive models use — each word can only attend to itself and earlier positions.

### Demo 5 — Multi-Head Attention

Three attention heads side by side on the same 10-word sentence, each with different hand-crafted projection matrices. One head tracks syntax (verbs ↔ nouns), one tracks coreference (pronouns → antecedents), one tracks positional proximity. Pick a word and see three completely different attention patterns for the same input.

---

## Files

| File | Purpose |
|---|---|
| `attention_playground.py` | Streamlit app — five interactive demos |
| `blog-post-attention.md` | Blog post (the narrative) |
| `ATTENTION_MATH_DEEP_DIVE.md` | Full mathematical treatment with worked examples |
| `requirements.txt` | Dependencies |

---

## What This Post Covers

| Concept | Blog section | Playground demo |
|---|---|---|
| RNN bottleneck | Problem 1: The Compressed Summary | Demo 1 |
| Dot-product & scaled attention | Problem 2: Word Order | Demo 2 |
| Cross-attention alignment | Problem 2: Tamil→English | Demo 3 |
| Self-attention | Self-Attention section | Demo 4 |
| Multi-head attention | Multi-Head Attention section | Demo 5 |
| Causal masking | Math Deep Dive §9 | Demo 4 (toggle) |

---

## Series Context

```
Post 1–5:  Foundations (perceptron → MLP → backprop → optimization → regularization)
Post 6:    CNNs (spatial structure)
Post 7:    BatchNorm & ResNets (enable depth)
Post 8:    RNNs & LSTMs (sequential structure)
Post 9:    Attention Mechanisms
Post 10:   Transformers (attention without recurrence)
```
