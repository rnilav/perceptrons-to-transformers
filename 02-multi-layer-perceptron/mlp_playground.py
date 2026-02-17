"""Interactive MLP Playground.

A Streamlit app for exploring how MLPs solve the XOR problem.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP, create_xor_network


def main():
    """Main application."""
    st.set_page_config(page_title="MLP Playground", layout="wide")
    
    st.title("🧠 Multi-Layer Perceptron: Solving XOR")
    st.markdown("""
    Explore how a simple 2-2-1 network with sigmoid activation solves the problem 
    that stumped single-layer perceptrons!
    
    **To exit:** Press `Ctrl+C` in the terminal and close this browser tab.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 XOR Solution", "🔗 Network Architecture", "⚖️ Perceptron vs MLP"])
    
    with tab1:
        xor_solution_tab()
    
    with tab2:
        network_architecture_tab()
    
    with tab3:
        comparison_tab()


def create_scaled_xor_network(scale: float = 1.0) -> MLP:
    """Create XOR network with scaled weights."""
    mlp = MLP(
        layer_sizes=[2, 2, 1],
        activations=['sigmoid', 'sigmoid'],
        random_state=42
    )
    
    # Base weights (realistic values)
    mlp.weights_[0] = np.array([
        [3.5, 3.5],
        [3.5, 3.5]
    ]) * scale
    mlp.biases_[0] = np.array([-1.5, -5.0]) * scale
    
    mlp.weights_[1] = np.array([
        [4.0],
        [-4.0]
    ]) * scale
    mlp.biases_[1] = np.array([-2.0]) * scale
    
    return mlp


def xor_solution_tab():
    """Tab 1: XOR Solution with interactive visualization."""
    st.header("🔀 The XOR Problem & Solution")
    
    st.markdown("""
    **Why XOR matters:** The XOR (exclusive-or) problem exposed the fundamental 
    limitation of single-layer perceptrons. A single perceptron cannot learn XOR 
    because it's not linearly separable - you cannot draw a single straight line 
    to separate the classes.
    
    But with a hidden layer, MLPs can solve it!
    """)
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Interactive Controls")
    
    weight_scale = st.sidebar.slider(
        "Weight Magnitude",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Adjust the magnitude of all weights. See how it affects the decision boundary!"
    )
    
    st.sidebar.markdown("""
    **What this does:**
    - **< 0.7:** Network struggles, boundary too weak
    - **0.8 - 1.5:** Smooth, curved boundary (ideal!)
    - **> 2.0:** Sharp, almost linear boundary
    """)
    
    show_boundary = st.sidebar.checkbox(
        "Show Decision Boundary",
        value=True,
        help="Visualize the curved boundary that separates the XOR classes"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### XOR Truth Table")
        st.markdown("""
        | Input 1 | Input 2 | Output |
        |---------|---------|--------|
        | 0       | 0       | 0      |
        | 0       | 1       | 1      |
        | 1       | 0       | 1      |
        | 1       | 1       | 0      |
        """)
        
        st.info("""
        **The Pattern:** Output is 1 when inputs are different, 0 when they're the same. 
        This creates a diagonal pattern that cannot be separated by a single straight line.
        """)
        
        st.markdown("---")
        st.markdown("### Network Results")
        
        # Create the XOR network with scaled weights
        mlp = create_scaled_xor_network(weight_scale)
        
        # Test on XOR data
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])
        output = mlp.predict(X_xor)
        predictions = (output >= 0.5).astype(int).flatten()
        
        # Show results table
        results_data = {
            "Input 1": X_xor[:, 0],
            "Input 2": X_xor[:, 1],
            "Expected": y_xor,
            "Predicted": predictions,
            "Raw Output": output.flatten().round(4)
        }
        
        st.dataframe(results_data, width='stretch')
        
        accuracy = np.mean(predictions == y_xor)
        if accuracy == 1.0:
            st.success(f"✅ **Accuracy: {accuracy:.0%}** - XOR solved perfectly!")
        elif accuracy >= 0.75:
            st.warning(f"⚠️ **Accuracy: {accuracy:.0%}** - Partially working, adjust weights!")
        else:
            st.error(f"❌ **Accuracy: {accuracy:.0%}** - Network failed, weights too weak!")
        
        st.markdown(f"""
        **Network Configuration:**
        - 2 inputs (x₁, x₂)
        - 2 hidden neurons (sigmoid)
        - 1 output neuron (sigmoid)
        - **Weight Scale:** {weight_scale:.1f}x
        """)
    
    with col2:
        st.markdown("### Decision Boundary Visualization")
        
        # XOR data points
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot decision boundary if enabled
        if show_boundary:
            # Create the XOR network with scaled weights
            mlp = create_scaled_xor_network(weight_scale)
            
            # Create mesh grid
            x_min, x_max = -0.5, 1.5
            y_min, y_max = -0.5, 1.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            
            # Predict on mesh
            Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary with color gradient
            contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
            # Plot the decision boundary line (where output = 0.5)
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='--')
            plt.colorbar(contour, ax=ax, label='Network Output')
        
        # Plot data points
        ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
                  c='blue', s=400, marker='o', edgecolors='k', linewidths=3, 
                  label='Class 0', zorder=3)
        ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
                  c='red', s=400, marker='s', edgecolors='k', linewidths=3, 
                  label='Class 1', zorder=3)
        
        ax.set_xlabel('Input 1', fontsize=14, weight='bold')
        ax.set_ylabel('Input 2', fontsize=14, weight='bold')
        
        # Dynamic title based on weight scale
        if weight_scale < 0.7:
            title_suffix = " (Weak Weights)"
        elif weight_scale > 2.0:
            title_suffix = " (Strong Weights)"
        else:
            title_suffix = " (Optimal)"
        
        ax.set_title(f'XOR: Curved Decision Boundary{title_suffix}', 
                    fontsize=16, weight='bold', pad=15)
        ax.legend(fontsize=13, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("### 📖 How to Read This")
        
        st.markdown("""
        **Color Gradient (Network Confidence):**
        - 🔴 Red = Network predicts class 1 (output close to 1.0)
        - 🔵 Blue = Network predicts class 0 (output close to 0.0)
        - 🟡 Yellow = Uncertain (output around 0.5)
        
        **Classification Rule:**
        - Output < 0.5 → Classify as Class 0
        - Output ≥ 0.5 → Classify as Class 1
        - The gradient shows confidence, not the classification itself
        
        **Black Dashed Line (Decision Boundary):**
        - Where network output = 0.5 (the threshold)
        - Points on one side → class 0, other side → class 1
        - **Key:** It's a smooth curve, not a straight line!
        
        **Try the slider** to see how weight magnitude affects the boundary shape!
        """)
        
        st.warning("""
        ⚠️ **Important Note:** These weights are **hand-crafted**, not learned! 
        
        This demo shows that MLPs **can** solve XOR with the right weights, but doesn't 
        show **how to find** those weights automatically. That's what backpropagation does, 
        which we'll cover in the next module!
        """)


def network_architecture_tab():
    """Tab 2: Network Architecture with weights and biases."""
    st.header("🔗 Network Architecture")
    
    st.markdown("""
    This shows the internal structure of the 2-2-1 MLP that solves XOR, 
    including all weights and biases.
    """)
    
    # Get weight scale from sidebar (shared state)
    weight_scale = st.session_state.get('weight_scale', 1.0)
    
    # Slider for this tab
    weight_scale = st.slider(
        "Weight Magnitude",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key='arch_weight_scale',
        help="Adjust to see how weights change"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### How It Works")
        
        st.markdown("""
        **Layer-by-Layer:**
        
        1. **Input Layer (Green):**
           - Receives x₁ and x₂ (the two inputs)
           - No computation, just passes values forward
        
        2. **Hidden Layer (Blue):**
           - 2 neurons with sigmoid activation
           - h₁ learns OR-like pattern (x₁ OR x₂)
           - h₂ learns AND-like pattern (x₁ AND x₂)
           - Each has weights from both inputs + bias
        
        3. **Output Layer (Red):**
           - 1 neuron with sigmoid activation
           - Combines h₁ and h₂ to produce XOR
           - Formula: OR AND NOT AND = XOR
        
        **Weights (Lines):**
        - Red = Positive (excitatory)
        - Blue = Negative (inhibitory)
        - Thickness = Magnitude
        
        **Biases (Yellow boxes):**
        - Shift the activation threshold
        - One per neuron (except inputs)
        """)
        
        st.info(f"""
        **Current Configuration:**
        - Weight Scale: {weight_scale:.1f}x
        - Total Parameters: 9 (6 weights + 3 biases)
        - These specific values solve XOR perfectly!
        
        ⚠️ **Important:** These weights are hand-crafted, not learned through training!
        """)
        
        st.markdown("---")
        st.markdown("### 🤔 How Were These Weights Found?")
        
        st.markdown("""
        **The Truth:** These weights were **manually designed** by understanding the problem:
        
        1. **Hidden neuron 1** needs to detect "at least one input is 1" (OR pattern)
           - Positive weights from both inputs
           - Negative bias to lower threshold
        
        2. **Hidden neuron 2** needs to detect "both inputs are 1" (AND pattern)
           - Positive weights from both inputs
           - Large negative bias to require both inputs
        
        3. **Output neuron** combines them: "OR but not AND" = XOR
           - Positive weight from h₁ (OR)
           - Negative weight from h₂ (AND)
        
        **The Problem:** For complex problems, we can't hand-craft weights like this!
        
        **The Solution:** Backpropagation (next module) automatically learns these weights 
        from data, so we don't have to figure them out manually.
        """)
    
    with col2:
        st.markdown("### Network Diagram")
        
        # Visualize the network with weights
        mlp = create_scaled_xor_network(weight_scale)
        
        fig_net, ax_net = plt.subplots(figsize=(10, 8))
        
        # Get the actual weights
        W1 = mlp.weights_[0]
        b1 = mlp.biases_[0]
        W2 = mlp.weights_[1]
        b2 = mlp.biases_[1]
        
        # Layer positions
        input_x = 0
        hidden_x = 3
        output_x = 6
        
        # Neuron positions
        input_y = [1, 3]
        hidden_y = [1, 3]
        output_y = [2]
        
        # Draw connections with weights
        for i, iy in enumerate(input_y):
            for j, hy in enumerate(hidden_y):
                weight = W1[i, j]
                color = 'red' if weight > 0 else 'blue'
                linewidth = min(abs(weight) / 1.5, 5)
                ax_net.plot([input_x, hidden_x], [iy, hy], 
                           color=color, linewidth=linewidth, alpha=0.6, zorder=1)
                mid_x = (input_x + hidden_x) / 2
                mid_y = (iy + hy) / 2
                ax_net.text(mid_x, mid_y, f'{weight:.2f}', 
                           fontsize=9, ha='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
        
        for j, hy in enumerate(hidden_y):
            weight = W2[j, 0]
            color = 'red' if weight > 0 else 'blue'
            linewidth = min(abs(weight) / 1.5, 5)
            ax_net.plot([hidden_x, output_x], [hy, output_y[0]], 
                       color=color, linewidth=linewidth, alpha=0.6, zorder=1)
            mid_x = (hidden_x + output_x) / 2
            mid_y = (hy + output_y[0]) / 2
            ax_net.text(mid_x, mid_y, f'{weight:.2f}', 
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
        
        # Draw neurons
        for i, y in enumerate(input_y):
            circle = plt.Circle((input_x, y), 0.3, color='lightgreen', 
                               ec='darkgreen', linewidth=2, zorder=3)
            ax_net.add_patch(circle)
            ax_net.text(input_x, y, f'x{i+1}', ha='center', va='center', 
                       fontsize=12, weight='bold')
        
        for j, y in enumerate(hidden_y):
            circle = plt.Circle((hidden_x, y), 0.3, color='lightblue', 
                               ec='darkblue', linewidth=2, zorder=3)
            ax_net.add_patch(circle)
            ax_net.text(hidden_x, y, f'h{j+1}', ha='center', va='center', 
                       fontsize=12, weight='bold')
            ax_net.text(hidden_x, y - 0.6, f'b={b1[j]:.2f}', 
                       ha='center', fontsize=9, style='italic',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='orange'))
        
        circle = plt.Circle((output_x, output_y[0]), 0.3, color='lightcoral', 
                           ec='darkred', linewidth=2, zorder=3)
        ax_net.add_patch(circle)
        ax_net.text(output_x, output_y[0], 'y', ha='center', va='center', 
                   fontsize=12, weight='bold')
        ax_net.text(output_x, output_y[0] - 0.6, f'b={b2[0]:.2f}', 
                   ha='center', fontsize=9, style='italic',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='orange'))
        
        # Layer labels
        ax_net.text(input_x, 4.2, 'Input\nLayer', ha='center', fontsize=11, weight='bold')
        ax_net.text(hidden_x, 4.2, 'Hidden Layer\n(sigmoid)', ha='center', fontsize=11, weight='bold')
        ax_net.text(output_x, 4.2, 'Output Layer\n(sigmoid)', ha='center', fontsize=11, weight='bold')
        
        # Legend
        ax_net.text(3, -0.5, 'Red = Positive | Blue = Negative | Thickness = Magnitude', 
                   ha='center', fontsize=10, style='italic')
        
        ax_net.set_xlim(-1, 7)
        ax_net.set_ylim(-1.5, 4.5)
        ax_net.axis('off')
        ax_net.set_title(f'2-2-1 Network (Scale: {weight_scale:.1f}x)', 
                        fontsize=14, weight='bold', pad=20)
        
        st.pyplot(fig_net)
        plt.close()


def comparison_tab():
    """Tab 3: Perceptron vs MLP comparison."""
    st.header("⚖️ Perceptron vs MLP")
    
    st.markdown("""
    See the fundamental differences between single-layer perceptrons and multi-layer perceptrons.
    """)
    
    st.markdown("### Side-by-Side Comparison")
    
    col1, col2 = st.columns(2)
    
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    with col1:
        st.markdown("#### 🔴 Perceptron (Single Layer)")
        
        # Create a simple perceptron visualization (binary decision)
        fig_p, ax_p = plt.subplots(figsize=(6, 6))
        
        # For perceptron, create binary regions (hard decision)
        xx_p, yy_p = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                                 np.linspace(-0.5, 1.5, 100))
        
        # Simple linear decision: x1 + x2 > 1 (one of many failed attempts)
        Z_p = (xx_p + yy_p > 1).astype(float)
        
        # Binary colormap (only 0 and 1, no gradient)
        ax_p.contourf(xx_p, yy_p, Z_p, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.4)
        ax_p.contour(xx_p, yy_p, Z_p, levels=[0.5], colors='orange', linewidths=3, linestyles='-')
        
        # Plot XOR points
        ax_p.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
                   c='blue', s=200, marker='o', edgecolors='k', linewidths=2, zorder=3)
        ax_p.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
                   c='red', s=200, marker='s', edgecolors='k', linewidths=2, zorder=3)
        
        ax_p.set_xlabel('Input 1', fontsize=11, weight='bold')
        ax_p.set_ylabel('Input 2', fontsize=11, weight='bold')
        ax_p.set_title('Hard Decision (0 or 1)', fontsize=12, weight='bold')
        ax_p.grid(True, alpha=0.3)
        ax_p.set_xlim(-0.5, 1.5)
        ax_p.set_ylim(-0.5, 1.5)
        
        st.pyplot(fig_p)
        plt.close()
        
        st.error("❌ **Accuracy: 50%** - Straight line can't separate XOR!")
        
        st.markdown("""
        **Characteristics:**
        - ⚠️ Only straight line boundaries
        - ⚠️ Binary output (0 or 1)
        - ⚠️ No uncertainty/probability
        - ⚠️ Hard cutoff at boundary
        - ❌ **Cannot solve XOR**
        """)
    
    with col2:
        st.markdown("#### 🟢 MLP (Multi-Layer)")
        
        # Show MLP with gradient
        fig_m, ax_m = plt.subplots(figsize=(6, 6))
        
        mlp_comp = create_scaled_xor_network(1.0)
        xx_m, yy_m = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                                 np.linspace(-0.5, 1.5, 100))
        Z_m = mlp_comp.predict(np.c_[xx_m.ravel(), yy_m.ravel()])
        Z_m = Z_m.reshape(xx_m.shape)
        
        # Gradient colormap (smooth probabilities)
        contour_m = ax_m.contourf(xx_m, yy_m, Z_m, levels=20, cmap='RdYlBu', alpha=0.6)
        ax_m.contour(xx_m, yy_m, Z_m, levels=[0.5], colors='black', linewidths=3, linestyles='--')
        plt.colorbar(contour_m, ax=ax_m, label='Probability')
        
        # Plot XOR points
        ax_m.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
                   c='blue', s=200, marker='o', edgecolors='k', linewidths=2, zorder=3)
        ax_m.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
                   c='red', s=200, marker='s', edgecolors='k', linewidths=2, zorder=3)
        
        ax_m.set_xlabel('Input 1', fontsize=11, weight='bold')
        ax_m.set_ylabel('Input 2', fontsize=11, weight='bold')
        ax_m.set_title('Soft Probability (0.0 to 1.0)', fontsize=12, weight='bold')
        ax_m.grid(True, alpha=0.3)
        ax_m.set_xlim(-0.5, 1.5)
        ax_m.set_ylim(-0.5, 1.5)
        
        st.pyplot(fig_m)
        plt.close()
        
        st.success("✅ **Accuracy: 100%** - Curved boundary solves XOR!")
        
        st.markdown("""
        **Characteristics:**
        - ✅ Curved boundaries possible
        - ✅ Probability output (0.0 to 1.0)
        - ✅ Shows uncertainty (gradient)
        - ✅ Smooth transitions
        - ✅ **Solves XOR perfectly**
        """)
    
    # Key differences
    st.markdown("---")
    st.markdown("### � Key Differences")
    
    col_diff1, col_diff2 = st.columns(2)
    
    with col_diff1:
        st.markdown("""
        **Perceptron (Single Layer):**
        
        🏗️ **Architecture:**
        - Input → Output (direct)
        - No hidden layers
        - Linear transformation only
        
        📏 **Decision Boundary:**
        - Can only draw straight lines
        - Hyperplane in n-dimensional space
        - Limited to linearly separable problems
        
        🎯 **Output:**
        - Hard decisions (0 or 1)
        - No confidence measure
        - Binary classification only
        
        ❌ **Limitation:**
        - Cannot solve XOR
        - Cannot learn non-linear patterns
        - Triggered "AI winter" in 1970s
        """)
    
    with col_diff2:
        st.markdown("""
        **MLP (Multi-Layer):**
        
        🏗️ **Architecture:**
        - Input → Hidden → Output
        - One or more hidden layers
        - Non-linear transformations
        
        📏 **Decision Boundary:**
        - Can draw curves and complex shapes
        - Transforms input space via hidden layers
        - Solves non-linearly separable problems
        
        🎯 **Output:**
        - Soft probabilities (0.0 to 1.0)
        - Shows confidence/uncertainty
        - Smooth gradient transitions
        
        ✅ **Breakthrough:**
        - Solves XOR perfectly
        - Learns complex non-linear patterns
        - Foundation for modern deep learning
        """)
    
    st.info("""
    💡 **The Breakthrough Insight:**
    
    The hidden layer doesn't just add complexity - it fundamentally changes what the network 
    can learn by transforming the input space into one where the problem becomes solvable.
    
    This discovery in the 1980s revived neural network research and paved the way for 
    modern deep learning, including the transformers that power today's AI systems.
    
    ⚠️ **What's Missing:** This demo shows MLPs **can** solve XOR, but not **how to learn** 
    the weights automatically. That's what backpropagation does - it's the algorithm that 
    automatically finds the right weights from training data. We'll cover that in the next module!
    """)


if __name__ == "__main__":
    main()
