# Post 10: Transformers — The Architecture Behind Modern AI

The final architecture in the series. No recurrence. No convolution. Just attention, feed-forward networks, and residual connections. The foundation of GPT, Claude, and every modern language model.

## Overview

The Transformer removes the sequential bottleneck entirely. Instead of processing tokens one at a time (like RNNs), every position is processed in parallel through layers of self-attention. Each layer refines the representation by letting every token attend to every other token.

This post builds a **decoder-only Transformer** (the GPT architecture) and trains it on Shakespeare to generate text character by character.

## Key Concepts

### 1. Decoder-Only Architecture
- No encoder-decoder split (that's for translation)
- Input: text so far → Output: next token probability
- Used by GPT, Claude, and all modern LLMs
- Simpler than full Transformer, same core principles

### 2. Core Components (All From Previous Posts)
- **Token + Position Embeddings**: Character → vector, position → vector, add them
- **Masked Self-Attention**: Each token attends to all previous tokens (Post 9)
- **Multi-Head Attention**: Multiple attention operations in parallel (Post 9)
- **Feed-Forward Network**: Two-layer MLP with GELU activation (Post 2)
- **Residual Connections**: Skip connections around every sublayer (Post 7)
- **Layer Normalization**: Stabilize activations between layers (Post 7/8)

### 3. Self-Supervised Learning
- No labels needed
- Training signal: predict the next character
- Model learns grammar, style, structure as a side effect
- Same principle scales to billion-parameter models

### 4. Causal Masking
- Token at position t can only see positions 1 through t
- No peeking at future tokens
- Implemented as upper-triangular mask in attention

## Installation

```bash
cd blogpost-perceptrons-to-transformers/10-transformer/v2
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch (first framework in the series!)
- Streamlit (for playground)
- NumPy, Matplotlib

## Interactive Playground

```bash
streamlit run transformer_playground.py
```

### What You'll See

**Two Pretrained Models:**
- **Small**: 112K parameters, d=64, 2 layers, context=64
  - Fast generation (~instant on CPU)
  - Rough output, basic structure
- **Large**: 826K parameters, d=128, 4 layers, context=128
  - Slower but better quality
  - Recognizable Shakespeare patterns

**Features:**
- Type any prompt (e.g., "ROMEO:", "To be or not")
- Adjust temperature (0.3 = conservative, 1.2 = creative)
- Generate 100-1000 characters
- Compare both models side by side

**Try These Prompts:**
```
ROMEO:
KING HENRY:
To be or not
First Citizen:
```

## Training Your Own Model

```bash
python train_models.py
```

This trains both models from scratch on `shakespeare.txt`:
- Small model: ~10 minutes on CPU
- Large model: ~50 minutes on CPU
- Models saved to `pretrained/gpt_small.pt` and `pretrained/gpt_large.pt`

**Training Details:**
- Character-level tokenization (no tokenizer needed)
- ~1MB Shakespeare text (~1M characters)
- Cross-entropy loss on next-character prediction
- Adam optimizer (lr=3e-4)
- Batch size: 32 sequences
- Context length: 64 (small) or 128 (large)

## Quick Start Example

```python
import torch
from transformer_playground import MiniGPT

# Load pretrained model
checkpoint = torch.load("pretrained/gpt_large.pt", map_location="cpu")
model = MiniGPT(
    vocab_size=checkpoint["config"]["vocab_size"],
    d_model=checkpoint["config"]["d_model"],
    n_heads=checkpoint["config"]["n_heads"],
    n_layers=checkpoint["config"]["n_layers"],
    ctx_len=checkpoint["config"]["ctx_len"],
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Encode prompt
chars = checkpoint["chars"]
encode = {ch: i for i, ch in enumerate(chars)}
prompt = "ROMEO:"
idx = torch.tensor([[encode[ch] for ch in prompt]])

# Generate
output = model.generate(idx, max_new=200, temperature=0.8)

# Decode
decode = {i: ch for i, ch in enumerate(chars)}
text = "".join([decode[i] for i in output[0].tolist()])
print(text)
```

## Architecture Diagram

```
Input: "To be or not to b"
         ↓
    [Token Embedding]      → each char becomes a 128-dim vector
         ↓
    [Position Embedding]   → add position signal (sin/cos)
         ↓
    ┌─────────────────┐
    │ Transformer     │
    │ Block × 4       │
    │                 │
    │ • Masked Self-  │    ← attend to all previous tokens
    │   Attention     │
    │ • Add & Norm    │    ← residual + layer norm
    │ • Feed-Forward  │    ← 2-layer MLP with GELU
    │ • Add & Norm    │    ← residual + layer norm
    └─────────────────┘
         ↓
    [Linear Head]          → 128-dim → vocab_size logits
         ↓
    [Softmax]              → probability distribution
         ↓
    Output: P(next char | all previous)
    Prediction: "e" (high probability)
```

## Comparison: Small vs Large Model

| Metric | Small (112K) | Large (826K) |
|--------|--------------|--------------|
| Parameters | 112,192 | 826,049 |
| d_model | 64 | 128 |
| Layers | 2 | 4 |
| Heads | 2 | 4 |
| Context | 64 chars | 128 chars |
| Training time | ~2 min | ~5 min |
| Final loss | ~1.8 | ~1.5 |
| Output quality | Rough structure | Recognizable Shakespeare |


## Troubleshooting

**"No pretrained models found"**
- Run `python train_models.py` first
- Models will be saved to `pretrained/` directory

**"Out of memory" during training**
- Reduce batch size in `train_models.py`
- Reduce context length
- Train only the small model

**Generated text is gibberish**
- Check that model trained successfully (loss should drop to ~1.5-1.8)
- Try lower temperature (0.5-0.7)
- Ensure prompt uses characters from training data

**Generation is too repetitive**
- Increase temperature (0.9-1.2)
- Model might be undertrained (train longer)

**Slow generation**
- Use the small model (112K params)
- Reduce generation length
- Consider GPU if available (edit code to use `device='cuda'`)

