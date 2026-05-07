"""
Train two GPT models on Shakespeare and save weights for the playground.

Small:  d=64,  2 layers, 4 heads, ctx=64   (~112K params)
Large:  d=128, 4 layers, 4 heads, ctx=128  (~826K params)

Run once. The playground loads the saved weights for instant inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import urllib.request

# Import model
import sys
sys.path.insert(0, os.path.dirname(__file__))
from transformer_playground import MiniGPT


def load_data():
    cache = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(cache):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(url, cache)
    with open(cache, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    encode = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([encode[ch] for ch in text], dtype=torch.long)
    return text, chars, encode, data


def train_model(name, vocab_size, d_model, n_heads, n_layers, ctx_len,
                data, steps, batch_size=32, lr=3e-4):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Config: d={d_model}, heads={n_heads}, layers={n_layers}, ctx={ctx_len}")

    model = MiniGPT(vocab_size, d_model, n_heads, n_layers, ctx_len)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for step in range(steps):
        ix = torch.randint(len(data) - ctx_len, (batch_size,))
        x = torch.stack([data[i:i + ctx_len] for i in ix])
        y = torch.stack([data[i + 1:i + ctx_len + 1] for i in ix])

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            print(f"  Step {step:5d}/{steps} | Loss: {loss.item():.3f} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return model


def main():
    text, chars, encode, data = load_data()
    vocab_size = len(chars)
    decode_map = {i: ch for i, ch in enumerate(chars)}

    save_dir = os.path.join(os.path.dirname(__file__), "pretrained")
    os.makedirs(save_dir, exist_ok=True)

    configs = [
        ("small", 64, 4, 2, 64, 3000),
        ("large", 128, 4, 4, 128, 5000),
    ]

    for name, d, h, nl, ctx, steps in configs:
        model = train_model(name, vocab_size, d, h, nl, ctx, data, steps)

        # Save model weights and config
        save_path = os.path.join(save_dir, f"gpt_{name}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "d_model": d,
                "n_heads": h,
                "n_layers": nl,
                "ctx_len": ctx,
            },
            "chars": chars,
        }, save_path)
        print(f"Saved: {save_path}")

        # Generate sample
        prompt = "ROMEO:"
        idx = torch.tensor([[encode[ch] for ch in prompt]])
        with torch.no_grad():
            out = model.generate(idx, 200, temperature=0.8)
        generated = "".join([decode_map[i] for i in out[0].tolist()])
        print(f"\nSample ({name}):\n{generated}\n")

    # Save char mapping separately for easy loading
    import json
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"chars": chars}, f)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
