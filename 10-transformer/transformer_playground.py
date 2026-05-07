"""
Transformer Playground — Generate Shakespeare with a pretrained tiny GPT.

Two pretrained models:
- Small (112K params, d=64, 2 layers)  — fast, rough output
- Large (826K params, d=128, 4 layers) — slower, recognizable Shakespeare

No training required. Type a prompt, generate instantly.

Run: streamlit run transformer_playground.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


# ── Model (same architecture used for training) ──────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, ctx_len=128, dropout=0.0):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            ctx = idx[:, -self.ctx_len:]
            logits = self(ctx)[:, -1, :] / temperature
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Load pretrained models ────────────────────────────────────────────────

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained")


@st.cache_resource
def load_model(name):
    path = os.path.join(PRETRAINED_DIR, f"gpt_{name}.pt")
    if not os.path.exists(path):
        return None, None, None

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    chars = checkpoint["chars"]

    model = MiniGPT(
        cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
        cfg["n_layers"], cfg["ctx_len"], dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    encode = {ch: i for i, ch in enumerate(chars)}
    decode_map = {i: ch for i, ch in enumerate(chars)}

    return model, encode, decode_map


# ── App ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Transformer Playground", page_icon="⚡", layout="wide")
st.title("⚡ Tiny GPT: Generate Shakespeare")
st.markdown(
    "Two pretrained decoder-only Transformers. "
    "Type a prompt, pick a model, generate instantly."
)

# Check if pretrained models exist
small_model, small_enc, small_dec = load_model("small")
large_model, large_enc, large_dec = load_model("large")

if small_model is None and large_model is None:
    st.error(
        "No pretrained models found. Run `python train_models.py` first "
        "to train and save the models."
    )
    st.stop()

# Controls
col1, col2 = st.columns([2, 1])

with col1:
    model_choice = st.selectbox(
        "Model",
        [m for m in ["Small (112K params, 2 layers)", "Large (826K params, 4 layers)"]
         if (m.startswith("Small") and small_model) or (m.startswith("Large") and large_model)],
    )

    prompt = st.text_input("Prompt", value="ROMEO:")
    gen_len = st.slider("Characters to generate", 100, 1000, 300, 50)
    temperature = st.slider("Temperature", 0.3, 1.5, 0.8, 0.1)

    generate_btn = st.button("Generate", type="primary", use_container_width=True)

with col2:
    st.markdown(
        "**Try these prompts:**\n"
        "- `ROMEO:` — dialogue\n"
        "- `KING HENRY:` — royal speech\n"
        "- `To be or not` — continuation\n"
        "- `First Citizen:` — commoner voice\n\n"
        "**Temperature:**\n"
        "- 0.3 = conservative, repetitive\n"
        "- 0.8 = balanced (default)\n"
        "- 1.2 = creative, risky"
    )

    if model_choice.startswith("Small"):
        st.info("**Small model:** 112K params, d=64, 2 layers, ctx=64. Fast but rough.")
    else:
        st.info("**Large model:** 826K params, d=128, 4 layers, ctx=128. Better quality.")

# Generate
if generate_btn:
    is_small = model_choice.startswith("Small")
    model = small_model if is_small else large_model
    encode = small_enc if is_small else large_enc
    decode_map = small_dec if is_small else large_dec

    # Encode prompt (skip unknown chars)
    idx = torch.tensor([[encode.get(ch, 0) for ch in prompt]])

    with torch.no_grad():
        out = model.generate(idx, gen_len, temperature=temperature)

    generated = "".join([decode_map[i] for i in out[0].tolist()])

    st.subheader("Generated Text")
    st.code(generated, language=None)

    st.caption(
        f"Model: {'small' if is_small else 'large'} | "
        f"Temperature: {temperature} | "
        f"Prompt: \"{prompt}\" | "
        f"Generated {gen_len} characters"
    )

    # Compare both models if both available
    if small_model and large_model and st.checkbox("Compare both models"):
        st.subheader("Side by Side")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Small (112K params)**")
            idx_s = torch.tensor([[small_enc.get(ch, 0) for ch in prompt]])
            with torch.no_grad():
                out_s = small_model.generate(idx_s, gen_len, temperature=temperature)
            st.code("".join([small_dec[i] for i in out_s[0].tolist()]), language=None)

        with c2:
            st.markdown("**Large (826K params)**")
            idx_l = torch.tensor([[large_enc.get(ch, 0) for ch in prompt]])
            with torch.no_grad():
                out_l = large_model.generate(idx_l, gen_len, temperature=temperature)
            st.code("".join([large_dec[i] for i in out_l[0].tolist()]), language=None)
