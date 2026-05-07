"""
Backpropagation Playground v2 — Interactive Streamlit App

Visual anchor: the loss curve dropping as the network learns.
Watch random weights converge to a solution, or get stuck in a local minimum.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "02-multi-layer-perceptron"))
from backprop import TrainableMLP


st.set_page_config(page_title="Backpropagation Playground", page_icon="🎓", layout="wide")

st.title("🎓 Backpropagation: Watch the Network Learn")
st.markdown(
    "No more hand-crafted weights. The network starts with random numbers "
    "and learns from its mistakes. Watch the loss curve drop."
)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# --- Controls ---

col_ctrl, col_info = st.columns([2, 1])

with col_ctrl:
    c1, c2 = st.columns(2)
    with c1:
        architecture = st.selectbox(
            "Architecture",
            {"2-2-1 (minimal, seed-sensitive)": [2, 2, 1], "2-4-1 (robust)": [2, 4, 1]},
            index=1,
        )
        arch = {"2-2-1 (minimal, seed-sensitive)": [2, 2, 1], "2-4-1 (robust)": [2, 4, 1]}[architecture]

        learning_rate = st.slider("Learning Rate", 0.1, 1.0, 0.5, 0.1)

    with c2:
        seed = st.number_input("Random Seed", 1, 999, 123)
        epochs = st.slider("Epochs", 500, 5000, 3000, 500)

    train = st.button("Train", type="primary", use_container_width=True)

with col_info:
    st.markdown(
        "**Try these:**\n"
        "1. Seed 123, 2-4-1, lr 0.5 → smooth convergence\n"
        "2. Seed 5, 2-2-1, lr 0.5 → stuck at 75% (local minimum)\n"
        "3. Same seed 5, switch to 2-4-1 → now it converges\n"
        "4. Learning rate 1.0 → watch the loss bounce"
    )

# --- Train and visualize ---

if train:
    mlp = TrainableMLP(
        layer_sizes=arch,
        activations=["sigmoid"] * (len(arch) - 1),
        learning_rate=learning_rate,
        random_state=int(seed),
    )

    progress = st.progress(0)
    chunk = max(1, epochs // 20)
    all_loss = []

    for start in range(0, epochs, chunk):
        n = min(chunk, epochs - start)
        history = mlp.train(X, Y, epochs=n, verbose=False)
        all_loss.extend(history["loss"])
        progress.progress((start + n) / epochs)

    progress.empty()

    preds = mlp.predict(X)
    preds_bin = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_bin == Y) * 100

    # --- Metrics ---

    m1, m2, m3 = st.columns(3)
    m1.metric("Final Loss", f"{all_loss[-1]:.4f}")
    m2.metric("Accuracy", f"{accuracy:.0f}%")
    m3.metric("Status", "Converged ✓" if accuracy == 100 else "Stuck ✗")

    if accuracy == 100:
        st.success("The network learned XOR from scratch.")
    elif accuracy >= 75:
        st.warning("Stuck in a local minimum. Try a different seed or larger architecture.")
    else:
        st.error("Training failed. Try increasing the learning rate or epochs.")

    # --- Plots ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(range(1, len(all_loss) + 1), all_loss, linewidth=2, color="#2E86AB")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (MSE)", fontsize=12)
    ax1.set_title("Loss Curve", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    contour = ax2.contourf(xx, yy, Z, levels=20, cmap="RdYlBu", alpha=0.6)
    ax2.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=3, linestyles="--")
    plt.colorbar(contour, ax=ax2, label="Network Output")

    y_flat = Y.flatten()
    ax2.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1],
                c="#0173B2", s=250, marker="o", edgecolors="white", linewidths=2, zorder=5,
                label="Class 0")
    ax2.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1],
                c="#DE8F05", s=250, marker="s", edgecolors="white", linewidths=2, zorder=5,
                label="Class 1")

    ax2.set_title(f"Decision Boundary ({accuracy:.0f}%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x₁", fontsize=12)
    ax2.set_ylabel("x₂", fontsize=12)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Predictions ---

    st.markdown("---")
    st.subheader("Predictions")
    cols = st.columns(4)
    for i, (col, x_i, y_true, y_pred) in enumerate(
        zip(cols, X, Y.flatten(), preds.flatten())
    ):
        with col:
            pred = 1 if y_pred > 0.5 else 0
            icon = "✓" if pred == y_true else "✗"
            st.write(f"[{x_i[0]}, {x_i[1]}] → {y_pred:.3f} → {pred} {icon}")

    st.caption(
        "The same algorithm that learned these 9 weights is what trains "
        "GPT-4's 1.76 trillion parameters. Forward pass, compute loss, "
        "backward pass, update weights. The scale changes, the principle doesn't."
    )
