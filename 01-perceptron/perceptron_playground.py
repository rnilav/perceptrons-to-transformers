"""
Perceptron Playground v2 - Interactive Streamlit App

Visual anchor: the decision boundary line.
Watch it cleanly separate AND/OR, then fail hopelessly on XOR.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directory so we can import the original perceptron
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perceptron import Perceptron


st.set_page_config(page_title="Perceptron Playground", page_icon="🧠", layout="wide")

st.title("🧠 Perceptron Playground")
st.markdown(
    "Train a perceptron on logic gates. Watch the decision boundary settle "
    "for AND and OR, then watch it fail on XOR."
)


# --- Dataset helpers ---

DATASETS = {
    "AND": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,0,1])),
    "OR":  (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,1])),
    "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0])),
    "NAND":(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([1,1,1,0])),
}


def plot_decision_boundary(X, y, model, title):
    """Plot data points, decision boundary line, and shaded regions."""
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()

    # Shaded decision regions
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale=[[0, "rgba(222,143,5,0.15)"], [1, "rgba(1,115,178,0.15)"]],
        showscale=False, hoverinfo="skip",
        contours=dict(start=0, end=1, size=1),
    ))

    # Decision boundary line
    params = model.get_params()
    w, b = params["weights"], params["bias"]
    if abs(w[1]) > 1e-8:
        x_line = np.array([x_min, x_max])
        y_line = -(w[0] * x_line + b) / w[1]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name="Decision Boundary",
            line=dict(color="red", width=3, dash="dash"),
        ))

    # Data points
    predictions = model.predict(X)
    colors = ["#DE8F05", "#0173B2"]
    for cls in [0, 1]:
        mask = y == cls
        correct = predictions[mask] == y[mask]
        # Correct points
        if np.any(correct):
            fig.add_trace(go.Scatter(
                x=X[mask][correct, 0], y=X[mask][correct, 1],
                mode="markers", name=f"Class {cls}",
                marker=dict(size=14, color=colors[cls],
                            line=dict(width=2, color="white")),
            ))
        # Misclassified points
        if np.any(~correct):
            fig.add_trace(go.Scatter(
                x=X[mask][~correct, 0], y=X[mask][~correct, 1],
                mode="markers", name="Misclassified",
                marker=dict(size=16, color="red", symbol="x",
                            line=dict(width=2)),
            ))

    fig.update_layout(
        title=title, xaxis_title="x₁", yaxis_title="x₂",
        template="plotly_white", height=480,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
    )
    return fig


def plot_errors(errors):
    """Plot training error curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(errors) + 1)), y=errors,
        mode="lines+markers",
        line=dict(color="#0173B2", width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch", yaxis_title="Misclassifications",
        template="plotly_white", height=280,
    )
    return fig


# --- Sidebar ---

st.sidebar.header("Configuration")

dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))

learning_rate = st.sidebar.slider(
    "Learning Rate", 0.01, 1.0, 0.1, 0.01,
    help="Step size for weight updates. Higher means faster but less stable.",
)

max_epochs = st.sidebar.slider(
    "Max Epochs", 10, 500, 100, 10,
    help="Maximum training iterations.",
)


# --- Tabs ---

tab_explore, tab_compare = st.tabs(["Explore", "AND vs XOR"])

with tab_explore:
    # --- Train ---

    X, y = DATASETS[dataset_name]
    model = Perceptron(learning_rate=learning_rate, n_iterations=max_epochs, random_state=42)
    model.fit(X, y)
    params = model.get_params()

    # --- Layout ---

    col_viz, col_info = st.columns([2, 1])

    with col_viz:
        st.plotly_chart(
            plot_decision_boundary(X, y, model, f"{dataset_name} Gate"),
            width='stretch',
        )
        st.plotly_chart(
            plot_errors(params["errors_per_epoch"]),
            width='stretch',
        )

    with col_info:
        st.subheader("Results")
        accuracy = model.score(X, y)
        st.metric("Accuracy", f"{accuracy:.0%}")
        st.metric("Epochs", params["n_epochs_trained"])

        if params["converged"]:
            st.success("Converged. The line found its place.")
        else:
            st.error("Did not converge. No single line can solve this.")

        st.markdown("---")
        st.subheader("Learned Parameters")
        st.write(f"**w₁:** {params['weights'][0]:.4f}")
        st.write(f"**w₂:** {params['weights'][1]:.4f}")
        st.write(f"**bias:** {params['bias']:.4f}")

        st.markdown("---")
        st.subheader("What to try")
        st.markdown(
            "1. Train on **AND** or **OR**. Watch the line settle.\n"
            "2. Switch to **XOR**. Watch it fail.\n"
            "3. Crank the learning rate to 1.0 on AND. Does it still converge?\n"
            "4. Notice: the error curve hits zero for AND, never for XOR."
        )

        st.markdown("---")
        st.caption(
            "The perceptron draws one straight line. "
            "If your problem lives on opposite sides of that line, it works. "
            "If not, no amount of training will help."
        )

with tab_compare:
    st.markdown(
        "**The same perceptron, two different problems.** "
        "AND is linearly separable. XOR is not. That's the whole story."
    )

    X_logic = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    model_and = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
    model_and.fit(X_logic, np.array([0, 0, 0, 1]))

    model_xor = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
    model_xor.fit(X_logic, np.array([0, 1, 1, 0]))

    col_and, col_xor = st.columns(2)

    with col_and:
        st.plotly_chart(
            plot_decision_boundary(
                X_logic, np.array([0, 0, 0, 1]), model_and, "AND: Line Settles ✓"
            ),
            width='stretch',
        )
        params_and = model_and.get_params()
        st.success(f"Converged in {params_and['n_epochs_trained']} epochs. Accuracy: 100%")

    with col_xor:
        st.plotly_chart(
            plot_decision_boundary(
                X_logic, np.array([0, 1, 1, 0]), model_xor, "XOR: Line Fails ✗"
            ),
            width='stretch',
        )
        params_xor = model_xor.get_params()
        st.error(f"100 epochs. Accuracy: {model_xor.score(X_logic, np.array([0,1,1,0])):.0%}. No line can solve this.")
