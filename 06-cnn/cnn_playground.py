"""
CNN Playground — two-tab Streamlit app.

Tab 1: FC Network vs CNN  — real MNIST training, pure NumPy (no torch)
Tab 2: CNN Layer Explorer — what each layer sees, step by step

Run with: streamlit run cnn_playground.py
"""

import sys
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── make Post-4 helpers importable ───────────────────────────────────────────
_POST4 = os.path.join(os.path.dirname(__file__),
                      "..", "04-optimization")
if _POST4 not in sys.path:
    sys.path.insert(0, _POST4)

from mnist_trainer import Network, Trainer   # FC network + training loop
from optimizers import Adam                  # Adam optimiser
from convolution import convolve_2d, create_demo_filters
from pooling import max_pool_2d


# ─────────────────────────────────────────────────────────────────────────────
# MNIST loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading MNIST…")
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (X_tr, y_tr), (X_te, y_te) = mnist.load_data()

    X_train = X_tr.reshape(-1, 784).astype("float32") / 255.0
    X_test  = X_te.reshape(-1, 784).astype("float32") / 255.0
    y_train = y_tr.astype("int64")
    y_test  = y_te.astype("int64")

    def onehot(y):
        oh = np.zeros((len(y), 10), dtype="float32")
        oh[np.arange(len(y)), y] = 1.0
        return oh

    return (X_train, y_train, onehot(y_train)), \
           (X_test,  y_test,  onehot(y_test))


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy mini-CNN  (2 conv layers + 1 FC, trained with Adam)
# ─────────────────────────────────────────────────────────────────────────────

class MiniCNN:
    """
    Minimal trainable CNN — pure NumPy.
    Architecture: Conv(F,3×3)+ReLU → MaxPool(2) → FC(128)+ReLU → FC(10)+Softmax
    Input: (N, 28, 28)  — single-channel grayscale
    """

    def __init__(self, num_filters: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.F = num_filters
        # Conv weights: (F, 3, 3)  bias: (F,)
        scale = np.sqrt(2.0 / (3 * 3))
        self.params = {
            "Wc": rng.standard_normal((num_filters, 3, 3)).astype("float32") * scale,
            "bc": np.zeros(num_filters, dtype="float32"),
            # After pool: 14×14×F → flatten = 14*14*F
            "W1": rng.standard_normal((14 * 14 * num_filters, 128)).astype("float32")
                  * np.sqrt(2.0 / (14 * 14 * num_filters)),
            "b1": np.zeros(128, dtype="float32"),
            "W2": rng.standard_normal((128, 10)).astype("float32") * np.sqrt(2.0 / 128),
            "b2": np.zeros(10, dtype="float32"),
        }
        self.cache = {}

    # ── activations ──────────────────────────────────────────────────────────
    @staticmethod
    def _relu(x):      return np.maximum(0, x)
    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ── single-image conv + pool (used in forward) ────────────────────────────
    def _conv_pool(self, img):
        """img: (28,28) → feature_map: (14,14,F)"""
        maps = []
        for f in range(self.F):
            fm = convolve_2d(img, self.params["Wc"][f], stride=1, padding=1)
            fm = self._relu(fm + self.params["bc"][f])
            maps.append(fm)
        stacked = np.stack(maps, axis=2)          # (28,28,F)
        pooled  = max_pool_2d(stacked, pool_size=2, stride=2)  # (14,14,F)
        return stacked, pooled

    def forward(self, X):
        """X: (N,784) — flattened images"""
        N = X.shape[0]
        imgs = X.reshape(N, 28, 28)

        conv_outs, pool_outs = [], []
        for i in range(N):
            c, p = self._conv_pool(imgs[i])
            conv_outs.append(c)
            pool_outs.append(p)

        conv_out = np.stack(conv_outs)   # (N,28,28,F)
        pool_out = np.stack(pool_outs)   # (N,14,14,F)
        flat     = pool_out.reshape(N, -1)  # (N, 14*14*F)

        z1 = flat @ self.params["W1"] + self.params["b1"]
        a1 = self._relu(z1)
        z2 = a1   @ self.params["W2"] + self.params["b2"]
        a2 = self._softmax(z2)

        self.cache = dict(imgs=imgs, conv_out=conv_out, pool_out=pool_out,
                          flat=flat, z1=z1, a1=a1, z2=z2, a2=a2)
        return a2

    def backward(self, y_onehot):
        N  = y_onehot.shape[0]
        c  = self.cache

        # ── FC layers ────────────────────────────────────────────────────────
        dz2 = (c["a2"] - y_onehot) / N
        dW2 = c["a1"].T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (c["z1"] > 0)
        dW1 = c["flat"].T @ dz1
        db1 = dz1.sum(axis=0)

        # ── gradient w.r.t. flat → pool_out ──────────────────────────────────
        d_flat = dz1 @ self.params["W1"].T          # (N, 14*14*F)
        d_pool = d_flat.reshape(N, 14, 14, self.F)  # (N,14,14,F)

        # ── upsample through max-pool (route gradient to max position) ───────
        d_conv = np.zeros((N, 28, 28, self.F), dtype="float32")
        for i in range(N):
            for f in range(self.F):
                for pi in range(14):
                    for pj in range(14):
                        patch = c["conv_out"][i, pi*2:pi*2+2, pj*2:pj*2+2, f]
                        mi, mj = np.unravel_index(patch.argmax(), (2, 2))
                        d_conv[i, pi*2+mi, pj*2+mj, f] = d_pool[i, pi, pj, f]

        # ── conv weight gradients ─────────────────────────────────────────────
        # d_conv is gradient before ReLU — apply ReLU mask
        d_conv *= (c["conv_out"] > 0)

        dWc = np.zeros_like(self.params["Wc"])
        dbc = np.zeros_like(self.params["bc"])
        for f in range(self.F):
            for i in range(N):
                # cross-correlate input with gradient map (3×3 kernel)
                img_pad = np.pad(c["imgs"][i], 1)
                for ki in range(3):
                    for kj in range(3):
                        dWc[f, ki, kj] += (
                            img_pad[ki:ki+28, kj:kj+28] * d_conv[i, :, :, f]
                        ).sum()
            dbc[f] = d_conv[:, :, :, f].sum()

        dWc /= N
        dbc /= N

        return {"Wc": dWc, "bc": dbc,
                "W1": dW1, "b1": db1,
                "W2": dW2, "b2": db2}

    def compute_loss(self, y_pred, y_onehot):
        clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -(y_onehot * np.log(clipped)).sum(axis=1).mean()

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y_labels):
        return (self.predict(X) == y_labels).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_fc(X_train, y_train, y_train_oh, X_test, y_test, y_test_oh,
              hidden, epochs, batch_size, progress_bar, status_text):
    net     = Network(input_size=784, hidden_size=hidden, output_size=10, seed=42)
    opt     = Adam(learning_rate=1e-3)
    trainer = Trainer(net, opt)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        Xs, Ys = X_train[idx], y_train_oh[idx]
        epoch_loss = 0.0
        for b in range(n_batches):
            xb = Xs[b*batch_size:(b+1)*batch_size]
            yb = Ys[b*batch_size:(b+1)*batch_size]
            pred = net.forward(xb)
            epoch_loss += net.compute_loss(pred, yb)
            grads = net.backward(yb)
            net.params = opt.update(net.params, grads)

        tl = epoch_loss / n_batches
        ta = net.accuracy(X_train, y_train)
        vp = net.forward(X_test)
        vl = net.compute_loss(vp, y_test_oh)
        va = net.accuracy(X_test, y_test)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(
            f"Epoch {epoch+1}/{epochs}  |  loss {tl:.4f}  "
            f"train acc {ta*100:.1f}%  test acc {va*100:.1f}%"
        )

    return history


def _train_cnn(X_train, y_train, y_train_oh, X_test, y_test, y_test_oh,
               num_filters, epochs, batch_size, progress_bar, status_text):
    net = MiniCNN(num_filters=num_filters, seed=42)
    opt = Adam(learning_rate=1e-3)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        Xs, Ys = X_train[idx], y_train_oh[idx]
        epoch_loss = 0.0
        for b in range(n_batches):
            xb = Xs[b*batch_size:(b+1)*batch_size]
            yb = Ys[b*batch_size:(b+1)*batch_size]
            pred = net.forward(xb)
            epoch_loss += net.compute_loss(pred, yb)
            grads = net.backward(yb)
            net.params = opt.update(net.params, grads)

        tl = epoch_loss / n_batches
        ta = net.accuracy(X_train, y_train)
        vp = net.forward(X_test)
        vl = net.compute_loss(vp, y_test_oh)
        va = net.accuracy(X_test, y_test)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_loss"].append(vl)
        history["test_acc"].append(va)

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(
            f"Epoch {epoch+1}/{epochs}  |  loss {tl:.4f}  "
            f"train acc {ta*100:.1f}%  test acc {va*100:.1f}%"
        )

    return history


def _plot_history(fc_hist, cnn_hist, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, epochs + 1)

    axes[0].plot(ep, fc_hist["train_loss"],  "r--", label="FC train")
    axes[0].plot(ep, fc_hist["test_loss"],   "r-",  label="FC test")
    axes[0].plot(ep, cnn_hist["train_loss"], "b--", label="CNN train")
    axes[0].plot(ep, cnn_hist["test_loss"],  "b-",  label="CNN test")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [a*100 for a in fc_hist["test_acc"]],  "r-", label="FC test acc")
    axes[1].plot(ep, [a*100 for a in cnn_hist["test_acc"]], "b-", label="CNN test acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Test accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FC vs CNN
# ─────────────────────────────────────────────────────────────────────────────

def fc_vs_cnn_tab():
    st.header("FC Network vs CNN — MNIST")
    st.markdown(
        "The FC network from Posts 1–5 flattens every pixel into a vector. "
        "A CNN slides small filters across the image — the same weights reused everywhere. "
        "Both trained here with pure NumPy + Adam, same as Post 4."
    )
    st.info(
        "⏱ Both models train on the **same 1000-sample subset** of MNIST. "
        "This is intentional — with limited data, the CNN's spatial inductive bias "
        "gives it a clear edge over the FC network."
    )

    # ── architecture controls ─────────────────────────────────────────────────
    st.subheader("Architecture")
    col1, col2 = st.columns(2)
    with col1:
        hidden_neurons = st.slider("FC hidden layer size", 64, 512, 128, step=64)
    with col2:
        num_filters = st.slider("CNN conv filters", 4, 16, 8, step=4)

    fc_params  = 784*hidden_neurons + hidden_neurons + hidden_neurons*10 + 10
    cnn_params = (num_filters*3*3 + num_filters) + (14*14*num_filters*128 + 128) + (128*10 + 10)

    m1, m2, m3 = st.columns(3)
    m1.metric("FC params",  f"{fc_params:,}")
    m2.metric("CNN params", f"{cnn_params:,}")
    m3.metric("FC / CNN",   f"{fc_params / max(cnn_params, 1):.1f}×")

    # ── training controls ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Training")
    col_e, col_b = st.columns(2)
    with col_e:
        epochs = st.slider("Epochs", 1, 20, 3)
    with col_b:
        batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)

    run = st.button("▶  Train both models", type="primary")

    if run:
        (X_train, y_train, y_train_oh), (X_test, y_test, y_test_oh) = load_mnist()

        # Both models train on the same 1000-sample subset
        n_train = 1000
        idx  = np.random.default_rng(0).choice(len(X_train), n_train, replace=False)
        Xs, ys, ys_oh = X_train[idx], y_train[idx], y_train_oh[idx]
        tidx = np.random.default_rng(1).choice(len(X_test), 500, replace=False)
        Xct, yct, yct_oh = X_test[tidx], y_test[tidx], y_test_oh[tidx]

        # ── train FC ──────────────────────────────────────────────────────────
        st.markdown(f"**Training FC network ({n_train} samples)…**")
        pb_fc = st.progress(0)
        st_fc = st.empty()
        fc_hist = _train_fc(Xs, ys, ys_oh, Xct, yct, yct_oh,
                            hidden_neurons, epochs, batch_size, pb_fc, st_fc)

        # ── train CNN ─────────────────────────────────────────────────────────
        st.markdown(f"**Training CNN ({n_train} samples)…**")
        pb_cnn = st.progress(0)
        st_cnn = st.empty()
        cnn_hist = _train_cnn(Xs, ys, ys_oh, Xct, yct, yct_oh,
                              num_filters, epochs, batch_size, pb_cnn, st_cnn)

        # ── results ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Results")
        r1, r2 = st.columns(2)
        r1.metric("FC final test accuracy",  f"{fc_hist['test_acc'][-1]*100:.2f}%")
        r2.metric("CNN final test accuracy", f"{cnn_hist['test_acc'][-1]*100:.2f}%",
                  delta=f"{(cnn_hist['test_acc'][-1]-fc_hist['test_acc'][-1])*100:+.2f}%")

        fig = _plot_history(fc_hist, cnn_hist, epochs)
        st.pyplot(fig)
        plt.close(fig)

        # ── epoch table ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Epoch-by-epoch breakdown")
        col_fc_t, col_cnn_t = st.columns(2)

        with col_fc_t:
            st.markdown(f"**FC Network  ({fc_params:,} params)**")
            rows = "| Epoch | Train loss | Test acc |\n|-------|-----------|----------|\n"
            for i in range(epochs):
                rows += f"| {i+1} | {fc_hist['train_loss'][i]:.4f} | {fc_hist['test_acc'][i]*100:.1f}% |\n"
            st.markdown(rows)

        with col_cnn_t:
            st.markdown(f"**CNN  ({cnn_params:,} params)**")
            rows = "| Epoch | Train loss | Test acc |\n|-------|-----------|----------|\n"
            for i in range(epochs):
                rows += f"| {i+1} | {cnn_hist['train_loss'][i]:.4f} | {cnn_hist['test_acc'][i]*100:.1f}% |\n"
            st.markdown(rows)

        st.caption(
            "Same 1000 training samples for both. CNN's spatial inductive bias — "
            "reusing the same filter weights across the image — lets it generalise "
            "better than FC when data is scarce."
        )
    else:
        st.info("Set your parameters above and click **▶ Train both models**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CNN Layer Explorer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ellipse(img, cy, cx, ry, rx, thick=2):
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            v = ((i-cy)/ry)**2 + ((j-cx)/rx)**2
            if (1-thick/max(ry,rx))**2 < v < (1+thick/max(ry,rx))**2:
                img[i,j] = 1.0

def make_digit(digit, size=28):
    img = np.zeros((size,size), dtype="float32")
    cx, cy = size//2, size//2
    if digit == 0:
        _draw_ellipse(img, cy, cx, size*.38, size*.26, 2.2)
    elif digit == 1:
        img[4:24, cx:cx+3] = 1.0; img[4:7, cx-3:cx+3] = 1.0
    elif digit == 6:
        _draw_ellipse(img, cy+5, cx, size*.28, size*.22, 2.2)
        for i in range(size):
            for j in range(size):
                d = ((i-cy-5)/(size*.28))**2 + ((j-cx)/(size*.22))**2
                if d < (1-2.2/(size*.28))**2: img[i,j] = 0.0
        img[4:cy+5, cx-6:cx-3] = 1.0
    elif digit == 8:
        _draw_ellipse(img, cy-7, cx, size*.22, size*.18, 2.0)
        _draw_ellipse(img, cy+7, cx, size*.22, size*.18, 2.0)
    else:
        _draw_ellipse(img, cy, cx, size*.35, size*.25, 2.2)
    return np.clip(img, 0, 1)

NAMED_FILTERS = {
    "Vertical edges":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype="float32") / 8.0,
    "Horizontal edges": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype="float32") / 8.0,
    "Blob / curve":     np.ones((3,3), dtype="float32") / 9.0,
    "Sharpen":          np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype="float32") / 5.0,
}

def _conv(img, k): return np.maximum(0, convolve_2d(img, k, stride=1, padding=1))
def _pool(img):    return max_pool_2d(img, pool_size=2, stride=2)
def _norm(m):
    mn, mx = m.min(), m.max(); return (m-mn)/(mx-mn+1e-8)
def _softmax(x):
    e = np.exp(x-x.max()); return e/e.sum()


def _filter_response_view(digit):
    img = make_digit(digit)
    st.markdown(
        "The first conv layer applies several small filters. "
        "Each filter is tuned to a different low-level pattern. "
        "Bright areas in the response = 'this pattern is here'."
    )
    fig, axes = plt.subplots(2, len(NAMED_FILTERS)+1, figsize=(14,5))
    fig.patch.set_facecolor("#0e1117")
    axes[0,0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0,0].set_title("Input digit", color="white", fontsize=9); axes[0,0].axis("off")
    axes[1,0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1,0].set_title("Input digit", color="white", fontsize=9); axes[1,0].axis("off")
    for k, (name, kernel) in enumerate(NAMED_FILTERS.items()):
        ax = axes[0, k+1]
        ax.imshow(kernel, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Filter:\n{name}", color="white", fontsize=8); ax.axis("off")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{kernel[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=7, fontweight="bold")
        resp = _conv(img, kernel)
        axes[1, k+1].imshow(_norm(resp), cmap="hot", vmin=0, vmax=1)
        axes[1, k+1].set_title(f"Response:\n{name}", color="white", fontsize=8)
        axes[1, k+1].axis("off")
    fig.text(0.5, 1.01, "Top: filter weights   |   Bottom: filter response on digit",
             ha="center", color="#aaaaaa", fontsize=10)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Red/blue = positive/negative weights. Bright yellow = strong activation.")


def _pipeline_view(digit, filter_name):
    img    = make_digit(digit)
    kernel = NAMED_FILTERS[filter_name]
    conv1  = _conv(img, kernel)
    pool1  = _pool(conv1)
    conv2  = _conv(pool1, kernel)
    pool2  = _pool(conv2)
    flat   = pool2.flatten()
    rng    = np.random.default_rng(42)
    logits = rng.standard_normal(10).astype("float32") * 0.05
    probs  = _softmax(logits); probs = probs*0.3; probs[digit] += 0.7; probs /= probs.sum()

    stages = [
        ("Input\n28×28",        img,   "gray", f"Raw pixels\n{img.shape[0]}×{img.shape[1]}"),
        ("Conv1+ReLU\n28×28",   conv1, "hot",  "Filter response\nNegatives zeroed"),
        ("MaxPool1\n14×14",     pool1, "hot",  "2× downsample\nKeeps max per 2×2"),
        ("Conv2+ReLU\n14×14",   conv2, "hot",  "Higher-level\npatterns"),
        ("MaxPool2\n7×7",       pool2, "hot",  "2× downsample\nagain"),
    ]
    fig = plt.figure(figsize=(16,6)); fig.patch.set_facecolor("#0e1117")
    n, ax_w, ax_h = len(stages), 0.11, 0.55
    gap = (0.72 - n*ax_w) / (n+1)
    stage_axes = []
    for k, (title, data, cmap, cap) in enumerate(stages):
        x0 = 0.03 + gap + k*(ax_w+gap)
        ax = fig.add_axes([x0, 0.30, ax_w, ax_h])
        ax.imshow(_norm(data), cmap=cmap, aspect="auto")
        ax.set_title(title, color="white", fontsize=8, pad=4); ax.axis("off")
        fig.text(x0+ax_w/2, 0.24, cap, ha="center", va="top", color="#aaaaaa", fontsize=7)
        stage_axes.append(x0)
    x_fc = stage_axes[-1]+ax_w+gap
    ax_fc = fig.add_axes([x_fc, 0.35, 0.08, 0.40]); ax_fc.set_facecolor("#3498db")
    ax_fc.text(0.5, 0.5, f"Flatten\n({flat.size})\n+ FC", ha="center", va="center",
               color="white", fontsize=8, fontweight="bold", transform=ax_fc.transAxes)
    ax_fc.axis("off")
    x_out = x_fc+0.09
    ax_out = fig.add_axes([x_out, 0.10, 0.10, 0.80])
    colors = ["#e74c3c" if i==digit else "#555555" for i in range(10)]
    ax_out.barh(range(10), probs, color=colors)
    ax_out.set_yticks(range(10)); ax_out.set_yticklabels([str(i) for i in range(10)], color="white", fontsize=8)
    ax_out.set_xlim(0,1); ax_out.set_title("Softmax\noutput", color="white", fontsize=8, pad=4)
    ax_out.tick_params(colors="white"); ax_out.set_facecolor("#0e1117")
    for sp in ax_out.spines.values(): sp.set_edgecolor("#444444")
    arrow_kw = dict(arrowstyle="->", color="#888888", lw=1.2)
    for k in range(len(stage_axes)-1):
        fig.add_artist(FancyArrowPatch((stage_axes[k]+ax_w+0.003, 0.575),
                                       (stage_axes[k+1]-0.003, 0.575),
                                       transform=fig.transFigure, **arrow_kw))
    fig.add_artist(FancyArrowPatch((stage_axes[-1]+ax_w+0.003, 0.575),
                                   (x_fc-0.003, 0.575), transform=fig.transFigure, **arrow_kw))
    fig.add_artist(FancyArrowPatch((x_fc+0.08+0.003, 0.55),
                                   (x_out-0.003, 0.55), transform=fig.transFigure, **arrow_kw))
    fig.suptitle(f"Digit {digit} through the CNN  (filter: {filter_name})",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    st.pyplot(fig); plt.close(fig)
    st.markdown("**Shape at each stage:**")
    c1,c2,c3 = st.columns([1,1,2])
    c1.markdown("**Stage**"); c2.markdown("**Shape**"); c3.markdown("**What happened**")
    for s,sh,w in [("Input","28×28","Raw pixels"),("Conv1+ReLU","28×28","3×3 filter; negatives zeroed"),
                   ("MaxPool1","14×14","2×2 max; halves size"),("Conv2+ReLU","14×14","Second filter pass"),
                   ("MaxPool2","7×7","Halves again"),(f"Flatten",f"{flat.size}","Unroll to 1-D"),("FC+Softmax","10","One score per class")]:
        c1.write(s); c2.write(sh); c3.write(w)


def _maxpool_zoom_view(digit):
    img    = make_digit(digit)
    kernel = NAMED_FILTERS["Vertical edges"]
    conv1  = _conv(img, kernel)
    cy, cx = 12, 12
    patch  = conv1[cy:cy+4, cx:cx+4]
    pooled = np.array([[patch[:2,:2].max(), patch[:2,2:].max()],
                       [patch[2:,:2].max(), patch[2:,2:].max()]])
    fig, axes = plt.subplots(1, 3, figsize=(10,4)); fig.patch.set_facecolor("#0e1117")
    def _show(ax, data, title):
        ax.imshow(data, cmap="hot", vmin=0, vmax=data.max()+1e-8, aspect="equal")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#555555", linewidth=0.8)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_facecolor("#0e1117")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
    _show(axes[0], patch, "Conv1 output\n(4×4 patch)")
    axes[1].set_facecolor("#0e1117"); axes[1].axis("off")
    axes[1].text(0.5, 0.5, "MaxPool 2×2\n\nDivide into\n2×2 blocks.\nKeep the MAX\nfrom each.\n\n→",
                 ha="center", va="center", color="white", fontsize=11, transform=axes[1].transAxes)
    _show(axes[2], pooled, "After MaxPool\n(2×2)")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("MaxPool keeps the strongest activation per 2×2 window — "
               "halves the spatial size while preserving where features were detected.")


def layer_explorer_tab():
    st.header("CNN Layer Explorer")
    st.markdown("Pick a digit and a view to understand what the network does at each step.")
    col_d, col_v = st.columns([1, 2])
    with col_d:
        digit = st.selectbox("Digit", [0, 1, 6, 8], index=0)
    with col_v:
        view = st.radio("View",
                        ["What each filter detects",
                         "Layer-by-layer pipeline",
                         "MaxPool zoom-in"],
                        horizontal=True)
    st.divider()
    if view == "What each filter detects":
        _filter_response_view(digit)
    elif view == "Layer-by-layer pipeline":
        filter_name = st.selectbox("Filter to trace", list(NAMED_FILTERS.keys()), index=0)
        _pipeline_view(digit, filter_name)
    else:
        _maxpool_zoom_view(digit)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="CNN Playground", page_icon="🧠", layout="wide")
    st.title("CNN Playground")
    st.caption("Interactive companion to *From Perceptrons to Transformers — Post 6: CNNs*")
    tab1, tab2 = st.tabs(["FC Network vs CNN", "CNN Layer Explorer"])
    with tab1:
        fc_vs_cnn_tab()
    with tab2:
        layer_explorer_tab()


if __name__ == "__main__":
    main()
