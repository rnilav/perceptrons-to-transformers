"""
Regularization Playground v2 — Interactive Streamlit App

Visual anchor: train vs test accuracy gap.
No regularization: gap widens. Dropout + weight decay: gap shrinks.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-optimization"))

from network_with_regularization import RegularizedNetwork, RegularizedTrainer
from optimizers import Adam
from mnist_trainer import load_mnist


st.set_page_config(page_title="Regularization Playground", page_icon="🎯", layout="wide")

st.title("🎯 Regularization Playground: Closing the Gap")
st.markdown(
    "Train with no regularization and watch the gap widen. "
    "Add dropout and weight decay, watch it shrink."
)


@st.cache_data
def get_mnist():
    return load_mnist()


(X_train, y_train, y_train_oh), (X_test, y_test, y_test_oh) = get_mnist()

# --- Controls ---

col_ctrl, col_tips = st.columns([2, 1])

with col_ctrl:
    c1, c2 = st.columns(2)
    with c1:
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.0, 0.1,
                                 help="0 = no dropout. 0.2-0.3 is typical.")
        weight_decay = st.select_slider("Weight Decay",
                                        options=[0.0, 0.0001, 0.001, 0.01],
                                        value=0.0,
                                        help="0 = no penalty. 0.0001-0.001 is typical.")
    with c2:
        epochs = st.number_input("Epochs", 5, 30, 10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.01, 0.001, 0.001)

    train = st.button("Train", type="primary", width='stretch')

with col_tips:
    st.markdown(
        "**Try these in order:**\n"
        "1. No regularization (defaults) → watch the gap\n"
        "2. Dropout 0.2 → gap shrinks\n"
        "3. Dropout 0.2 + weight decay 0.0001 → gap shrinks more\n"
        "4. Dropout 0.5 → too much, accuracy drops"
    )

# --- Train and visualize ---

if train:
    network = RegularizedNetwork(
        input_size=784, hidden_size=128, output_size=10,
        dropout_rate=dropout_rate, seed=42,
    )
    optimizer = Adam(learning_rate)
    trainer = RegularizedTrainer(network, optimizer, weight_decay=weight_decay)

    progress = st.progress(0)
    status = st.empty()

    # Train epoch by epoch for progress updates
    history = None
    for ep in range(1, int(epochs) + 1):
        history = trainer.train(
            X_train, y_train_oh, y_train,
            X_test, y_test_oh, y_test,
            epochs=1, batch_size=64, verbose=False,
        )
        progress.progress(ep / int(epochs))
        status.text(f"Epoch {ep}/{int(epochs)}")

    progress.empty()
    status.empty()

    train_accs = [a * 100 for a in history["train_acc"]]
    test_accs = [a * 100 for a in history["test_acc"]]
    gap = train_accs[-1] - test_accs[-1]

    # --- Metrics ---

    m1, m2, m3 = st.columns(3)
    m1.metric("Train Accuracy", f"{train_accs[-1]:.1f}%")
    m2.metric("Test Accuracy", f"{test_accs[-1]:.1f}%")
    m3.metric("Gap", f"{gap:.1f}%")

    if gap < 1.5:
        st.success("Small gap. The network is generalizing well.")
    elif gap < 3:
        st.warning("Moderate gap. Some overfitting. Try adding or increasing regularization.")
    else:
        st.error("Large gap. The network is memorizing. Add dropout or weight decay.")

    # --- Plot ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ep_list = list(range(1, len(train_accs) + 1))

    ax1.plot(ep_list, train_accs, linewidth=2.5, color="#d62728", label="Train", marker="o", markersize=6)
    ax1.plot(ep_list, test_accs, linewidth=2.5, color="#2E86AB", label="Test", marker="s", markersize=6)
    ax1.fill_between(ep_list, test_accs, train_accs, alpha=0.15, color="#d62728", label="Gap")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Train vs Test Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    losses = history["train_loss"]
    ax2.plot(list(range(1, len(losses) + 1)), losses, linewidth=2.5, color="#2ca02c", marker="^", markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width='stretch')
    plt.close(fig)

    st.markdown("---")
    reg_desc = []
    if dropout_rate > 0:
        reg_desc.append(f"dropout={dropout_rate}")
    if weight_decay > 0:
        reg_desc.append(f"weight_decay={weight_decay}")
    reg_str = ", ".join(reg_desc) if reg_desc else "none"

    st.caption(
        f"Regularization: {reg_str}. "
        f"The shaded area between the curves is the generalization gap. "
        f"Smaller gap = better generalization."
    )
else:
    st.info("Configure settings and click Train to begin.")
