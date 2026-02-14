"""
Perceptron Playground - Interactive Streamlit App

An educational tool for exploring how perceptrons learn and their limitations.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from perceptron import Perceptron

# Page configuration
st.set_page_config(
    page_title="Perceptron Playground",
    page_icon="🧠",
    layout="wide"
)

# Title and introduction
st.title("🧠 Perceptron Playground")
st.markdown("""
Explore the first artificial neuron! Train a perceptron on different datasets 
and see how it learns decision boundaries. Discover why it can solve AND and OR 
but fails on XOR.
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Learning parameters
learning_rate = st.sidebar.slider(
    "Learning Rate (α)",
    min_value=0.001,
    max_value=1.0,
    value=0.1,
    step=0.001,
    help="Step size for weight updates. Higher = faster but less stable."
)

n_iterations = st.sidebar.slider(
    "Max Iterations",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Maximum number of training epochs."
)

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["AND", "OR", "XOR", "NAND", "Random (Separable)", "Random (Non-separable)"],
    help="Choose a logical function or random data."
)

# Generate dataset button
if st.sidebar.button("🔄 Generate New Random Data"):
    st.session_state.random_seed = np.random.randint(0, 10000)

# Initialize random seed
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

# Helper functions
def generate_dataset(choice, seed=42):
    """Generate dataset based on choice."""
    if choice == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        return X, y, "AND: Output 1 only when both inputs are 1"
    
    elif choice == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        return X, y, "OR: Output 1 when at least one input is 1"
    
    elif choice == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        return X, y, "XOR: Output 1 when inputs are different (NOT linearly separable!)"
    
    elif choice == "NAND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([1, 1, 1, 0])
        return X, y, "NAND: Output 0 only when both inputs are 1"
    
    elif choice == "Random (Separable)":
        np.random.seed(seed)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y, "Random linearly separable data"
    
    else:  # Random (Non-separable)
        np.random.seed(seed)
        X = np.random.randn(50, 2)
        # Create non-separable data (XOR-like pattern)
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
        return X, y, "Random non-separable data (XOR pattern)"

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """Plot decision boundary with data points."""
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add contour for decision regions
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale=[[0, 'rgba(222, 143, 5, 0.3)'], [1, 'rgba(1, 115, 178, 0.3)']],
        showscale=False,
        hoverinfo='skip',
        contours=dict(start=0, end=1, size=1)
    ))
    
    # Get predictions for data points
    predictions = model.predict(X)
    correct = predictions == y
    
    # Add data points (correct predictions)
    colors = ['#DE8F05', '#0173B2']
    for class_value in [0, 1]:
        mask = (y == class_value) & correct
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'Class {class_value} (correct)',
                marker=dict(
                    size=12,
                    color=colors[class_value],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                showlegend=True
            ))
    
    # Add misclassified points
    mask_wrong = ~correct
    if np.any(mask_wrong):
        fig.add_trace(go.Scatter(
            x=X[mask_wrong, 0],
            y=X[mask_wrong, 1],
            mode='markers',
            name='Misclassified',
            marker=dict(
                size=14,
                color='red',
                symbol='x',
                line=dict(width=2)
            ),
            showlegend=True
        ))
    
    # Add decision boundary line
    params = model.get_params()
    w = params['weights']
    b = params['bias']
    
    if w[1] != 0:
        x_line = np.array([x_min, x_max])
        y_line = -(w[0] * x_line + b) / w[1]
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Decision Boundary',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='x₁',
        yaxis_title='x₂',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

# Main content
col1, col2 = st.columns([2, 1])

# Generate dataset
X, y, description = generate_dataset(dataset_choice, st.session_state.random_seed)

with col1:
    st.subheader("📊 Visualization")
    
    # Train model
    model = Perceptron(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        random_state=42
    )
    model.fit(X, y)
    
    # Plot decision boundary
    fig = plot_decision_boundary(X, y, model, f"{dataset_choice} Function")
    st.plotly_chart(fig, use_container_width=True)
    
    # Training progress
    params = model.get_params()
    errors = params['errors_per_epoch']
    
    fig_errors = go.Figure()
    fig_errors.add_trace(go.Scatter(
        x=list(range(1, len(errors) + 1)),
        y=errors,
        mode='lines+markers',
        name='Errors',
        line=dict(color='#0173B2', width=2),
        marker=dict(size=6)
    ))
    
    fig_errors.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Number of Misclassifications',
        template='plotly_white',
        height=300
    )
    
    st.plotly_chart(fig_errors, use_container_width=True)

with col2:
    st.subheader("📈 Metrics")
    
    # Get model parameters
    params = model.get_params()
    accuracy = model.score(X, y)
    
    # Display metrics
    st.metric("Accuracy", f"{accuracy:.1%}")
    st.metric("Epochs Trained", params['n_epochs_trained'])
    st.metric("Final Errors", params['errors_per_epoch'][-1])
    
    # Convergence status
    if params['converged']:
        st.success("✅ Converged!")
    else:
        st.warning("⚠️ Did not converge")
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("🔧 Model Parameters")
    st.write(f"**Weight 1 (w₁):** {params['weights'][0]:.4f}")
    st.write(f"**Weight 2 (w₂):** {params['weights'][1]:.4f}")
    st.write(f"**Bias (b):** {params['bias']:.4f}")
    
    # Decision boundary equation
    w = params['weights']
    b = params['bias']
    st.markdown("**Decision Boundary:**")
    st.latex(f"{w[0]:.2f}x_1 + {w[1]:.2f}x_2 + {b:.2f} = 0")
    
    st.markdown("---")
    
    # Dataset info
    st.subheader("📝 Dataset Info")
    st.info(description)
    st.write(f"**Samples:** {len(X)}")
    st.write(f"**Features:** {X.shape[1]}")
    st.write(f"**Classes:** {len(np.unique(y))}")

# Expandable sections
with st.expander("📚 How to Use This App"):
    st.markdown("""
    1. **Choose a dataset** from the sidebar (AND, OR, XOR, etc.)
    2. **Adjust learning rate** to see how it affects convergence speed
    3. **Adjust max iterations** if the model doesn't converge
    4. **Observe the decision boundary** (red dashed line)
    5. **Check if it converges** - XOR will never converge!
    
    **Key Insights:**
    - AND, OR, NAND are **linearly separable** → perceptron converges
    - XOR is **not linearly separable** → perceptron fails
    - Higher learning rate → faster convergence (but can be unstable)
    - Lower learning rate → slower but more stable convergence
    """)

with st.expander("🧮 Mathematical Details"):
    st.markdown("""
    ### Perceptron Model
    
    The perceptron computes:
    """)
    st.latex(r"\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)")
    
    st.markdown("""
    Where σ is the step function:
    """)
    st.latex(r"\sigma(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}")
    
    st.markdown("""
    ### Learning Rule
    
    For each training example:
    """)
    st.latex(r"w \leftarrow w + \alpha \cdot (y - \hat{y}) \cdot x")
    st.latex(r"b \leftarrow b + \alpha \cdot (y - \hat{y})")
    
    st.markdown("""
    Where α is the learning rate.
    """)

with st.expander("❓ Why XOR Fails"):
    st.markdown("""
    ### The XOR Problem
    
    XOR (exclusive OR) outputs 1 when inputs are different:
    
    | x₁ | x₂ | XOR |
    |----|----|----|
    | 0  | 0  | 0  |
    | 0  | 1  | 1  |
    | 1  | 0  | 1  |
    | 1  | 1  | 0  |
    
    **No single straight line can separate the 1s from the 0s!**
    
    This is called **linear inseparability** and is the fundamental limitation 
    of single-layer perceptrons.
    
    **Solution:** Use multilayer perceptrons (MLPs) with hidden layers, which 
    can learn non-linear decision boundaries.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for AI education | 
    <a href='https://github.com/yourusername/perceptrons-to-transformers'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
