"""
Attention Playground — Post 9: Attention Mechanisms

Five concept demos, one narrative. No training loops, no waiting.
Every interaction updates instantly so you can *see* the math.

Run:
    streamlit run attention_playground.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Playground", layout="wide", page_icon="🔍")

# ── shared helpers ────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


ACCENT = "#3a86ff"
WARM   = "#ff6b35"
TEAL   = "#2ec4b6"
GRAY   = "#8d99ae"

# ── title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Attention Playground")
st.markdown(
    "Five interactive demos that follow the blog post narrative. "
    "Every slider updates instantly — no training, no waiting. Just the math, visualised."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 1 — The Bottleneck: Why Compression Loses Information
# ══════════════════════════════════════════════════════════════════════════════

st.header("1 · The Bottleneck")
st.markdown(
    "An RNN encoder reads a sentence word by word, updating a hidden state at each step. "
    "At the end, only the *final* hidden state is passed to the decoder. "
    "The problem: early tokens get exponentially overwritten. "
    "Drag the sentence length to see how little of the first word survives."
)

col1a, col1b = st.columns([1, 2])

with col1a:
    seq_len = st.slider("Sentence length (tokens)", 3, 50, 10, key="bn_seq")
    # Show how much of token 1 survives
    decay_rate = 0.85
    token1_survival = decay_rate ** (seq_len - 1)
    st.metric("First token's contribution", f"{token1_survival:.1%}",
              help="How much of the first token's information remains in the final hidden state")
    st.caption(
        f"After {seq_len - 1} steps of overwriting, the first token contributes "
        f"just **{token1_survival:.1%}** to the final vector — "
        f"regardless of whether the hidden state has 16 or 1024 dimensions. "
        "The decay is exponential, not a capacity problem."
    )

with col1b:
    # Simulate the effective contribution of each token to the final hidden state.
    # In a vanilla RNN, token t's contribution decays as λ^(n-t) where λ < 1.
    decay_rate = 0.85  # typical spectral radius
    positions = np.arange(1, seq_len + 1)
    contributions = decay_rate ** (seq_len - positions)
    contributions = contributions / contributions.max()

    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [ACCENT] * len(positions)
    colors[-1] = WARM
    # Fade early tokens that contribute < 10%
    for i, c in enumerate(contributions):
        if c < 0.10:
            colors[i] = "#d0d0d0"
    bars = ax.bar(positions, contributions, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Token position in the sentence", fontsize=11)
    ax.set_ylabel("Contribution to final hidden state", fontsize=11)
    ax.set_title("What survives in the encoder's final vector", fontsize=12, weight="bold")
    ax.set_ylim(0, 1.15)
    # Smart x-ticks
    if seq_len <= 20:
        ax.set_xticks(positions)
    elif seq_len <= 35:
        ax.set_xticks(positions[::2])
    else:
        ax.set_xticks(positions[::5])
    ax.text(positions[-1], 1.04, "latest",
            ha="center", fontsize=8, color=WARM, weight="bold")
    ax.text(positions[0], contributions[0] + 0.04,
            f"{contributions[0]:.0%}", ha="center", fontsize=8, color="#999")
    ax.axhline(y=0.10, color="#ccc", linestyle="--", linewidth=0.8)
    ax.text(seq_len * 0.6, 0.12, "10% threshold", fontsize=7, color="#aaa")
    ax.spines[["top", "right"]].set_visible(False)
    st.image(fig_to_buf(fig))

st.info(
    "💡 **With attention**, the decoder can look back at *any* bar directly — "
    "no compression, no decay. That's the fix."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 2 — Dot-Product Attention: Pick a Word, See Where It Looks
# ══════════════════════════════════════════════════════════════════════════════

st.header("2 · Dot-Product Attention")
st.markdown(
    "Pick a sentence and a query word. The demo computes dot-product scores "
    "between the query and every other word, applies softmax, and shows you "
    "the attention weights. Below the main chart, a second panel shows why "
    "√d scaling matters at high dimensions."
)

# Pre-built sentences with hand-crafted embeddings that produce meaningful patterns
SENTENCES = {
    "The cat sat on the mat": {
        "tokens": ["The", "cat", "sat", "on", "the", "mat"],
        "embeds": np.array([
            [0.1, 0.9, 0.0, 0.2],   # The  — determiner
            [0.8, 0.1, 0.6, 0.3],   # cat  — noun/animal
            [0.3, 0.2, 0.9, 0.7],   # sat  — verb
            [0.0, 0.5, 0.1, 0.8],   # on   — preposition
            [0.1, 0.9, 0.0, 0.2],   # the  — determiner
            [0.7, 0.2, 0.5, 0.4],   # mat  — noun/object
        ], dtype=np.float64),
    },
    "She gave him the book yesterday": {
        "tokens": ["She", "gave", "him", "the", "book", "yesterday"],
        "embeds": np.array([
            [0.9, 0.1, 0.3, 0.5],   # She       — pronoun/subject
            [0.2, 0.3, 0.8, 0.6],   # gave      — verb
            [0.8, 0.2, 0.4, 0.5],   # him       — pronoun/object
            [0.1, 0.9, 0.0, 0.2],   # the       — determiner
            [0.6, 0.3, 0.5, 0.1],   # book      — noun
            [0.0, 0.7, 0.1, 0.9],   # yesterday — adverb/time
        ], dtype=np.float64),
    },
    "The report the client requested is ready": {
        "tokens": ["The", "report", "the", "client", "requested", "is", "ready"],
        "embeds": np.array([
            [0.1, 0.9, 0.0, 0.2],   # The
            [0.7, 0.2, 0.5, 0.3],   # report    — noun
            [0.1, 0.9, 0.0, 0.2],   # the
            [0.8, 0.1, 0.4, 0.6],   # client    — noun/agent
            [0.3, 0.2, 0.9, 0.5],   # requested — verb
            [0.2, 0.6, 0.3, 0.7],   # is        — copula
            [0.4, 0.5, 0.7, 0.8],   # ready     — adjective
        ], dtype=np.float64),
    },
}

col2a, col2b = st.columns([1, 2])

with col2a:
    sent_choice = st.selectbox("Sentence", list(SENTENCES.keys()), key="dp_sent")
    data = SENTENCES[sent_choice]
    tokens = data["tokens"]
    embeds = data["embeds"]
    query_word = st.selectbox("Query word (the one doing the looking)", tokens, key="dp_qw")
    d_k = embeds.shape[1]

with col2b:
    qi = tokens.index(query_word)
    q = embeds[qi]
    scores_raw = embeds @ q
    scores_scaled = scores_raw / np.sqrt(d_k)
    weights = softmax(scores_scaled)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), gridspec_kw={"height_ratios": [1, 1.4]})

    # Top: scaled scores as bar chart
    ax0 = axes[0]
    bar_colors = [WARM if i == qi else ACCENT for i in range(len(tokens))]
    ax0.bar(tokens, scores_scaled, color=bar_colors, edgecolor="white")
    ax0.set_ylabel("Score", fontsize=10)
    ax0.set_title("Scaled dot-product scores (÷ √d)", fontsize=11, weight="bold")
    ax0.spines[["top", "right"]].set_visible(False)

    # Bottom: attention weights as horizontal colour bar + numbers
    ax1 = axes[1]
    ax1.set_xlim(-0.5, len(tokens) - 0.5)
    ax1.set_ylim(0, 1)
    cmap = plt.cm.Blues
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        rect = plt.Rectangle((i - 0.45, 0.15), 0.9, 0.55,
                              facecolor=cmap(w / weights.max()),
                              edgecolor="white", linewidth=2)
        ax1.add_patch(rect)
        ax1.text(i, 0.42, f"{w:.1%}", ha="center", va="center",
                 fontsize=12, weight="bold",
                 color="white" if w > 0.4 * weights.max() else "#333")
        ax1.text(i, 0.82, tok, ha="center", va="center", fontsize=11)
        if i == qi:
            ax1.text(i, 0.02, "← query", ha="center", fontsize=8, color=WARM, style="italic")
    ax1.set_title(f"Attention weights when \"{query_word}\" looks at the sentence",
                  fontsize=11, weight="bold")
    ax1.axis("off")

    fig.tight_layout(pad=1.5)
    st.image(fig_to_buf(fig))

# ── Why √d scaling matters — shown at realistic dimensions ───────────────
st.subheader("Why √d scaling matters")
st.markdown(
    "At d=4 (above), scaling barely changes anything. "
    "But real models use d=64 or d=512. Use the slider to see what happens "
    "to the same set of scores as dimension grows."
)

dim_choice = st.slider("Simulated embedding dimension (d)", 4, 512, 64, step=4, key="dp_dim")

# Use fixed raw scores that represent a realistic scenario:
# one strong match, one moderate, one weak
raw_scores_base = np.array([2.1, 0.8, -0.3])
score_labels = ["strong match", "moderate", "weak"]

# At dimension d, dot products scale as √d, so we simulate that
scale_factor = np.sqrt(dim_choice / 4.0)  # relative to our base at d=4
raw_at_dim = raw_scores_base * scale_factor

unscaled_weights = softmax(raw_at_dim)
scaled_weights   = softmax(raw_at_dim / np.sqrt(dim_choice))

fig2, (ax_un, ax_sc) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

ax_un.bar(score_labels, unscaled_weights, color="#e05252", edgecolor="white")
for i, w in enumerate(unscaled_weights):
    ax_un.text(i, w + 0.02, f"{w:.1%}", ha="center", fontsize=10, weight="bold")
ax_un.set_title(f"Without scaling (d={dim_choice})", fontsize=11, weight="bold")
ax_un.set_ylim(0, 1.15)
ax_un.set_ylabel("Attention weight", fontsize=10)
ax_un.spines[["top", "right"]].set_visible(False)

ax_sc.bar(score_labels, scaled_weights, color=TEAL, edgecolor="white")
for i, w in enumerate(scaled_weights):
    ax_sc.text(i, w + 0.02, f"{w:.1%}", ha="center", fontsize=10, weight="bold")
ax_sc.set_title(f"With ÷ √{dim_choice} scaling", fontsize=11, weight="bold")
ax_sc.set_ylim(0, 1.15)
ax_sc.spines[["top", "right"]].set_visible(False)

fig2.tight_layout(pad=2.0)
st.image(fig_to_buf(fig2))

if dim_choice >= 64:
    st.caption(
        f"At d={dim_choice}, unscaled softmax collapses to near-100% on the strongest match — "
        "gradients die and the model stops learning. Scaling keeps the distribution smooth."
    )
else:
    st.caption(
        f"At d={dim_choice}, the difference is small. Try sliding to d=256 or d=512 to see the collapse."
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 3 — Word Reordering: The Tamil → English Alignment Matrix
# ══════════════════════════════════════════════════════════════════════════════

st.header("3 · Word Reordering (Cross-Attention)")
st.markdown(
    "The blog post describes translating Tamil (SOV) to English (SVO). "
    "The attention alignment matrix shows *which source word the decoder looks at* "
    "when generating each target word. Below are two matrices side by side: "
    "a naive left-to-right alignment vs. the real attention pattern."
)

# Fixed example from the blog post
tamil_gloss    = ["by-tmrw", "that", "report", "send", "can-you?"]
english_tokens = ["Can", "you", "send", "the", "report", "by", "tomorrow"]

# Real alignment: which Tamil position each English word attends to most
real_focus    = [4, 4, 3, 1, 2, 0, 0]  # from the blog post

n_eng = len(english_tokens)
n_tam = len(tamil_gloss)


def build_alignment(focus_list):
    """Build a soft alignment matrix peaked at the given positions."""
    mat = np.zeros((n_eng, n_tam))
    for i, f in enumerate(focus_list):
        raw = np.full(n_tam, -1.0)
        raw[f] = 3.0
        for j in range(n_tam):
            if j != f:
                raw[j] = -1.0 + 0.5 * (1.0 - abs(j - f) / n_tam)
        mat[i] = softmax(raw)
    return mat


# Naive: each English word just looks at the Tamil word in the same position (clamped)
naive_focus = [min(i, n_tam - 1) for i in range(n_eng)]

align_naive = build_alignment(naive_focus)
align_real  = build_alignment(real_focus)

st.markdown(
    "**How to read:** each row is an English word being generated. "
    "The bright cell in that row shows which Tamil word the decoder is looking at."
)

fig, (ax_naive, ax_real) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

for ax, alignment, title, cmap in [
    (ax_naive, align_naive, "Naive (left-to-right)", "Greys"),
    (ax_real,  align_real,  "Learned attention (reordered)", "YlOrRd"),
]:
    im = ax.imshow(alignment, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_tam))
    ax.set_xticklabels(tamil_gloss, fontsize=9)
    ax.set_yticks(range(n_eng))
    ax.set_yticklabels(english_tokens, fontsize=10)
    ax.set_xlabel("Tamil source", fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold")
    for i in range(n_eng):
        for j in range(n_tam):
            v = alignment[i, j]
            color = "white" if v > 0.5 else "#333"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=8, color=color, weight="bold" if v > 0.4 else "normal")

ax_naive.set_ylabel("English target", fontsize=10, weight="bold")
fig.tight_layout(pad=2.0)
st.image(fig_to_buf(fig))

st.info(
    '💡 The left matrix follows Tamil word order — "Can" looks at "by-tmrw", which is wrong. '
    "The right matrix shows what attention actually learns: "
    '"Can" jumps to "can-you?" (position 5), "send" jumps to "send" (position 4), '
    '"tomorrow" reaches back to "by-tmrw" (position 1). '
    "The non-diagonal pattern *is* the reordering."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 4 — Self-Attention: Every Word Sees Every Other Word
# ══════════════════════════════════════════════════════════════════════════════

st.header("4 · Self-Attention Heatmap")
st.markdown(
    "In self-attention, every word attends to every other word *in the same sentence*. "
    "Pick a sentence, then click any word to see its full attention distribution. "
    "The heatmap shows the complete n×n attention matrix."
)

SELF_ATT_SENTENCES = {
    "The cat sat on the mat": {
        "tokens": ["The", "cat", "sat", "on", "the", "mat"],
        "embeds": np.array([
            [0.1, 0.9, 0.0, 0.2],
            [0.8, 0.1, 0.6, 0.3],
            [0.3, 0.2, 0.9, 0.7],
            [0.0, 0.5, 0.1, 0.8],
            [0.1, 0.9, 0.0, 0.2],
            [0.7, 0.2, 0.5, 0.4],
        ], dtype=np.float64),
    },
    "The report the client requested is ready": {
        "tokens": ["The", "report", "the", "client", "requested", "is", "ready"],
        "embeds": np.array([
            [0.1, 0.9, 0.0, 0.2],
            [0.7, 0.2, 0.5, 0.3],
            [0.1, 0.9, 0.0, 0.2],
            [0.8, 0.1, 0.4, 0.6],
            [0.3, 0.2, 0.9, 0.5],
            [0.2, 0.6, 0.3, 0.7],
            [0.4, 0.5, 0.7, 0.8],
        ], dtype=np.float64),
    },
    "The cat sat because it was tired": {
        "tokens": ["The", "cat", "sat", "because", "it", "was", "tired"],
        "embeds": np.array([
            [0.1, 0.9, 0.0, 0.2],
            [0.8, 0.1, 0.6, 0.3],
            [0.3, 0.2, 0.9, 0.7],
            [0.0, 0.4, 0.2, 0.9],
            [0.7, 0.1, 0.5, 0.4],
            [0.2, 0.6, 0.3, 0.7],
            [0.5, 0.3, 0.8, 0.6],
        ], dtype=np.float64),
    },
}

col4a, col4b = st.columns([1, 2])

with col4a:
    sa_sent = st.selectbox("Sentence", list(SELF_ATT_SENTENCES.keys()), key="sa_sent")
    sa_data = SELF_ATT_SENTENCES[sa_sent]
    sa_tokens = sa_data["tokens"]
    sa_embeds = sa_data["embeds"]
    sa_highlight = st.selectbox("Highlight row for word", sa_tokens, key="sa_hl")
    sa_use_causal = st.toggle("Apply causal mask (decoder-style)", value=False, key="sa_causal")

with col4b:
    d = sa_embeds.shape[1]
    n = len(sa_tokens)
    scores = (sa_embeds @ sa_embeds.T) / np.sqrt(d)

    if sa_use_causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores[mask] = -1e9

    attn = softmax(scores, axis=-1)
    hi = sa_tokens.index(sa_highlight)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                              gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: full heatmap
    ax0 = axes[0]
    im = ax0.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax0.set_xticks(range(n))
    ax0.set_xticklabels(sa_tokens, fontsize=9, rotation=45, ha="right")
    ax0.set_yticks(range(n))
    ax0.set_yticklabels(sa_tokens, fontsize=9)
    ax0.set_xlabel("Key (attended to)", fontsize=10)
    ax0.set_ylabel("Query (doing the looking)", fontsize=10)
    title = "Self-Attention Matrix"
    if sa_use_causal:
        title += " (causal mask)"
    ax0.set_title(title, fontsize=11, weight="bold")
    # Annotate
    for i in range(n):
        for j in range(n):
            v = attn[i, j]
            color = "white" if v > 0.5 * attn.max() else "#333"
            ax0.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color=color)
    # Highlight selected row
    ax0.add_patch(plt.Rectangle((-0.5, hi - 0.5), n, 1,
                                 fill=False, edgecolor=WARM, linewidth=2.5))
    fig.colorbar(im, ax=ax0, shrink=0.75)

    # Right: bar chart for the highlighted word
    ax1 = axes[1]
    colors = [WARM if i == hi else ACCENT for i in range(n)]
    ax1.barh(sa_tokens[::-1], attn[hi][::-1], color=colors[::-1], edgecolor="white")
    ax1.set_xlabel("Attention weight", fontsize=10)
    ax1.set_title(f'Where "{sa_highlight}" looks', fontsize=11, weight="bold")
    ax1.set_xlim(0, attn[hi].max() * 1.25)
    for i, (tok, w) in enumerate(zip(sa_tokens[::-1], attn[hi][::-1])):
        ax1.text(w + 0.005, i, f"{w:.0%}", va="center", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.5)
    st.image(fig_to_buf(fig))

    if sa_use_causal:
        st.caption(
            "With the causal mask, each word can only attend to itself and earlier words. "
            "Notice the lower-triangular pattern — future positions are zeroed out."
        )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 5 — Multi-Head Attention: Different Heads See Different Things
# ══════════════════════════════════════════════════════════════════════════════

st.header("5 · Multi-Head Attention")
st.markdown(
    "Each attention head uses different projection matrices, so it learns to "
    "notice different relationships. This demo shows 3 heads side by side on "
    "the same sentence — each with a different \"personality\"."
)

MH_SENTENCE = "The cat sat on the mat because it was tired"
MH_TOKENS = MH_SENTENCE.split()
MH_N = len(MH_TOKENS)

# Three heads with hand-crafted projection matrices that produce distinct patterns.
# Each head projects the base embeddings into a 2D Q/K space.
MH_BASE = np.array([
    [0.1, 0.9, 0.0, 0.2],   # The
    [0.8, 0.1, 0.6, 0.3],   # cat
    [0.3, 0.2, 0.9, 0.7],   # sat
    [0.0, 0.5, 0.1, 0.8],   # on
    [0.1, 0.9, 0.0, 0.2],   # the
    [0.7, 0.2, 0.5, 0.4],   # mat
    [0.0, 0.4, 0.2, 0.9],   # because
    [0.7, 0.1, 0.5, 0.4],   # it
    [0.2, 0.6, 0.3, 0.7],   # was
    [0.5, 0.3, 0.8, 0.6],   # tired
], dtype=np.float64)

# Head 1: "Syntactic" — projects onto dims that make nouns attend to verbs
W_q1 = np.array([[0.8, 0.0], [0.0, 0.2], [0.5, 0.0], [0.0, 0.5]])
W_k1 = np.array([[0.0, 0.5], [0.2, 0.0], [0.8, 0.0], [0.5, 0.0]])

# Head 2: "Coreference" — projects onto dims that make pronouns attend to nouns
W_q2 = np.array([[0.9, 0.0], [0.0, 0.1], [0.0, 0.8], [0.0, 0.3]])
W_k2 = np.array([[0.9, 0.0], [0.0, 0.1], [0.3, 0.0], [0.0, 0.2]])

# Head 3: "Positional" — projects onto dims that make nearby words attend to each other
W_q3 = np.array([[0.1, 0.0], [0.5, 0.0], [0.0, 0.3], [0.0, 0.9]])
W_k3 = np.array([[0.1, 0.0], [0.5, 0.0], [0.0, 0.3], [0.0, 0.9]])

heads_info = [
    ("Head 1 · Syntactic",    W_q1, W_k1, "Oranges",  "Verbs ↔ nouns"),
    ("Head 2 · Coreference",  W_q2, W_k2, "Purples",  "Pronouns → antecedents"),
    ("Head 3 · Positional",   W_q3, W_k3, "Greens",   "Nearby words"),
]

col5a, col5b = st.columns([1, 3])

with col5a:
    mh_word = st.selectbox("Highlight word", MH_TOKENS, index=7, key="mh_word")
    mh_hi = MH_TOKENS.index(mh_word)
    st.markdown(
        f'Showing where **"{mh_word}"** looks in each head. '
        "Notice how different heads produce different attention patterns."
    )

with col5b:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (title, Wq, Wk, cmap_name, desc) in zip(axes, heads_info):
        Q = MH_BASE @ Wq
        K = MH_BASE @ Wk
        d_head = Q.shape[1]
        scores = (Q @ K.T) / np.sqrt(d_head)
        attn = softmax(scores, axis=-1)

        im = ax.imshow(attn, cmap=cmap_name, vmin=0, vmax=attn.max(), aspect="auto")
        ax.set_xticks(range(MH_N))
        ax.set_xticklabels(MH_TOKENS, fontsize=7, rotation=55, ha="right")
        ax.set_yticks(range(MH_N))
        ax.set_yticklabels(MH_TOKENS, fontsize=7)
        ax.set_title(f"{title}\n({desc})", fontsize=9, weight="bold")
        # Highlight row
        ax.add_patch(plt.Rectangle((-0.5, mh_hi - 0.5), MH_N, 1,
                                    fill=False, edgecolor=WARM, linewidth=2))

    fig.suptitle(
        f'Where "{mh_word}" looks — three heads, three perspectives',
        fontsize=12, weight="bold", y=1.02
    )
    fig.tight_layout(pad=1.0)
    st.image(fig_to_buf(fig))

# Bar chart comparison for the highlighted word
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 2.8), sharey=True)
for ax, (title, Wq, Wk, _, _) in zip(axes2, heads_info):
    Q = MH_BASE @ Wq
    K = MH_BASE @ Wk
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    attn = softmax(scores, axis=-1)
    row = attn[mh_hi]
    colors = [WARM if i == mh_hi else ACCENT for i in range(MH_N)]
    ax.bar(range(MH_N), row, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(MH_N))
    ax.set_xticklabels(MH_TOKENS, fontsize=7, rotation=45, ha="right")
    ax.set_title(title.split("·")[1].strip(), fontsize=9, weight="bold")
    ax.set_ylim(0, row.max() * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    # Label top-2
    top2 = np.argsort(row)[-2:]
    for idx in top2:
        ax.text(idx, row[idx] + 0.01, f"{row[idx]:.0%}",
                ha="center", fontsize=8, weight="bold")

axes2[0].set_ylabel("Attention weight", fontsize=9)
fig2.suptitle(f'"{mh_word}" — attention distribution per head', fontsize=11, weight="bold")
fig2.tight_layout(pad=1.0)
st.image(fig_to_buf(fig2))

st.info(
    "💡 A single head can only capture one relationship. "
    "Multi-head attention captures syntax, coreference, and proximity *simultaneously* — "
    "then combines them through the output projection."
)
st.markdown("---")

# ── footer ────────────────────────────────────────────────────────────────────
st.caption("All computations are pure NumPy — no training, no GPU, no waiting.")
