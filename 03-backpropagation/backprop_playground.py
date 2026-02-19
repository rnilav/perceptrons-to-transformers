"""Interactive Backpropagation Playground.

A Streamlit app for exploring how neural networks learn through backpropagation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backprop import TrainableMLP


# ============================================================================
# Helper Functions for Visualizations
# ============================================================================

def plot_loss_curve(loss_history: list, ax: plt.Axes = None) -> plt.Figure:
    """
    Plot training loss curve over epochs.
    
    Args:
        loss_history: List of loss values for each epoch
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, linewidth=2, color='#2E86AB', label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, weight='bold')
    ax.set_title('Training Loss Over Time', fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return fig


def plot_decision_boundary(mlp: TrainableMLP, X: np.ndarray, y: np.ndarray, 
                          ax: plt.Axes = None, title: str = "Decision Boundary") -> plt.Figure:
    """
    Plot decision boundary for a 2D classification problem.
    
    Args:
        mlp: Trained MLP model
        X: Input data of shape (n_samples, 2)
        y: Target labels of shape (n_samples,) or (n_samples, 1)
        ax: Optional matplotlib axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Flatten y if needed
    if y.ndim == 2:
        y = y.flatten()
    
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
    ax.scatter(X[y == 0, 0], X[y == 0, 1], 
              c='blue', s=400, marker='o', edgecolors='k', linewidths=3, 
              label='Class 0', zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], 
              c='red', s=400, marker='s', edgecolors='k', linewidths=3, 
              label='Class 1', zorder=3)
    
    ax.set_xlabel('Input 1', fontsize=12, weight='bold')
    ax.set_ylabel('Input 2', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return fig


def plot_network_with_gradients(mlp: TrainableMLP, weight_gradients: list = None,
                                bias_gradients: list = None, ax: plt.Axes = None) -> plt.Figure:
    """
    Plot network architecture diagram with gradient information.
    
    Args:
        mlp: MLP model
        weight_gradients: Optional list of weight gradient arrays
        bias_gradients: Optional list of bias gradient arrays
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Get network structure
    layer_sizes = mlp.layer_sizes
    n_layers = len(layer_sizes)
    
    # Layer positions
    layer_x_positions = np.linspace(0, 6, n_layers)
    
    # Calculate neuron positions for each layer
    neuron_positions = []
    for layer_size in layer_sizes:
        if layer_size == 1:
            positions = [2.0]  # Center single neuron
        else:
            positions = np.linspace(0.5, 3.5, layer_size)
        neuron_positions.append(positions)
    
    # Draw connections with weights (and gradients if provided)
    for layer_idx in range(n_layers - 1):
        x_from = layer_x_positions[layer_idx]
        x_to = layer_x_positions[layer_idx + 1]
        
        for i, y_from in enumerate(neuron_positions[layer_idx]):
            for j, y_to in enumerate(neuron_positions[layer_idx + 1]):
                weight = mlp.weights_[layer_idx][i, j]
                
                # Color based on weight sign
                color = 'red' if weight > 0 else 'blue'
                
                # Line width based on weight magnitude
                linewidth = min(abs(weight) / 1.5, 5)
                
                # If gradients provided, adjust alpha based on gradient magnitude
                alpha = 0.6
                if weight_gradients is not None:
                    grad = weight_gradients[layer_idx][i, j]
                    # Normalize gradient for visualization
                    max_grad = np.max(np.abs(weight_gradients[layer_idx]))
                    if max_grad > 0:
                        alpha = 0.3 + 0.7 * (abs(grad) / max_grad)
                
                ax.plot([x_from, x_to], [y_from, y_to], 
                       color=color, linewidth=linewidth, alpha=alpha, zorder=1)
                
                # Add weight label
                mid_x = (x_from + x_to) / 2
                mid_y = (y_from + y_to) / 2
                label = f'{weight:.2f}'
                if weight_gradients is not None:
                    grad = weight_gradients[layer_idx][i, j]
                    label += f'\n∇={grad:.3f}'
                
                ax.text(mid_x, mid_y, label, 
                       fontsize=8, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.8))
    
    # Draw neurons
    layer_colors = ['lightgreen', 'lightblue', 'lightcoral']
    layer_edge_colors = ['darkgreen', 'darkblue', 'darkred']
    layer_names = ['Input', 'Hidden', 'Output']
    
    for layer_idx, (x_pos, y_positions) in enumerate(zip(layer_x_positions, neuron_positions)):
        color_idx = min(layer_idx, len(layer_colors) - 1)
        
        for neuron_idx, y_pos in enumerate(y_positions):
            circle = plt.Circle((x_pos, y_pos), 0.25, 
                               color=layer_colors[color_idx], 
                               ec=layer_edge_colors[color_idx], 
                               linewidth=2, zorder=3)
            ax.add_patch(circle)
            
            # Neuron label
            if layer_idx == 0:
                label = f'x{neuron_idx + 1}'
            elif layer_idx == n_layers - 1:
                label = 'y'
            else:
                label = f'h{neuron_idx + 1}'
            
            ax.text(x_pos, y_pos, label, ha='center', va='center', 
                   fontsize=10, weight='bold', zorder=4)
            
            # Add bias labels (except for input layer)
            if layer_idx > 0:
                bias = mlp.biases_[layer_idx - 1][neuron_idx]
                bias_label = f'b={bias:.2f}'
                
                if bias_gradients is not None:
                    grad = bias_gradients[layer_idx - 1][neuron_idx]
                    bias_label += f'\n∇={grad:.3f}'
                
                ax.text(x_pos, y_pos - 0.5, bias_label, 
                       ha='center', fontsize=8, style='italic',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                                edgecolor='orange', alpha=0.8))
        
        # Layer labels
        layer_name = layer_names[color_idx] if color_idx < len(layer_names) else f'Layer {layer_idx}'
        ax.text(x_pos, 4.5, f'{layer_name}\nLayer', ha='center', 
               fontsize=11, weight='bold')
    
    # Legend
    legend_text = 'Red = Positive | Blue = Negative | Thickness = Magnitude'
    if weight_gradients is not None:
        legend_text += ' | Opacity = Gradient Magnitude'
    ax.text(3, -0.8, legend_text, ha='center', fontsize=10, style='italic')
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1.5, 5)
    ax.axis('off')
    ax.set_title('Network Architecture', fontsize=14, weight='bold', pad=20)
    
    return fig


def main():
    """Main application."""
    st.set_page_config(page_title="Backpropagation Playground", layout="wide")
    
    st.title("🎓 Backpropagation: How Neural Networks Learn")
    st.markdown("""
    Explore how neural networks automatically learn weights through backpropagation 
    and gradient descent - no more hand-crafted weights!
    
    **To exit:** Press `Ctrl+C` in the terminal and close this browser tab.
    """)
    
    tab1, tab2 = st.tabs([
        "🎯 Training Visualization",
        "🔄 Gradient Flow"
    ])
    with tab1:
        training_visualization_tab()
    
    with tab2:
        gradient_flow_tab()


def training_visualization_tab():
    """Tab 1: Training Visualization for XOR problem."""
    st.header("🔀 Watch the Network Learn XOR")
    
    st.markdown("""
    **The Big Difference:** In the previous module, we hand-crafted weights to solve XOR. 
    Now watch the network **learn** those weights automatically through backpropagation!
    """)
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Create two columns for controls and info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Training Controls")
        
        # Learning rate slider
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Controls the step size for weight updates. Too high = unstable, too low = slow convergence."
        )
        
        # Epochs slider
        epochs = st.slider(
            "Number of Epochs",
            min_value=100,
            max_value=5000,
            value=3000,
            step=100,
            help="How many times the network sees the training data. More epochs = more learning time."
        )
        
        # Architecture selector
        architecture_options = {
            "2-2-1 (Minimal)": [2, 2, 1],
            "2-4-1 (Robust)": [2, 4, 1],
            "2-8-1 (Overkill)": [2, 8, 1]
        }
        architecture_choice = st.selectbox(
            "Network Architecture",
            options=list(architecture_options.keys()),
            index=1,  # Default to 2-4-1
            help="Number of neurons in each layer. More neurons = more capacity but slower training."
        )
        architecture = architecture_options[architecture_choice]
        
        # Random seed selector
        seed_options = {
            "Seed 123 (Recommended)": 123,
            "Seed 42 (May struggle with 2-2-1)": 42,
            "Seed 456 (Good)": 456,
            "Seed 789 (May struggle with 2-2-1)": 789,
            "Random": None
        }
        seed_choice = st.selectbox(
            "Weight Initialization",
            options=list(seed_options.keys()),
            index=0,  # Default to 123
            help="Random seed for weight initialization. Different seeds can lead to different results!"
        )
        random_seed = seed_options[seed_choice]
        
        # Train button
        train_button = st.button("🚀 Train Network", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("💡 Key Insights")
        
        st.info("""
        **Automatic Learning**
        
        Unlike Post 2 where we hand-crafted weights, this network learns them automatically through backpropagation!
        """)
        
        st.warning("""
        **Learning Rate Effects**
        
        • 0.01-0.1: Slow but stable
        • 0.3: Balanced (recommended)
        • 0.5-0.7: Fast but may overshoot
        • 1.0+: Often unstable
        """)
        
        st.success("""
        **Architecture Tradeoffs**
        
        • 2-2-1: Minimal, but seed-sensitive
        • 2-4-1: More robust (recommended)
        • 2-8-1: Overkill for XOR, but very robust
        """)
    
    # Training execution
    if train_button:
        with st.spinner("🔄 Training network... Watch the loss decrease!"):
            # Create and train the network
            mlp = TrainableMLP(
                layer_sizes=architecture,
                activations=['sigmoid'] * (len(architecture) - 1),
                learning_rate=learning_rate,
                random_state=random_seed
            )
            
            # Train with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # We'll train in chunks to show progress
            chunk_size = max(1, epochs // 20)  # 20 updates
            history = {'loss': []}
            
            for chunk_start in range(0, epochs, chunk_size):
                chunk_epochs = min(chunk_size, epochs - chunk_start)
                chunk_history = mlp.train(X_xor, y_xor, epochs=chunk_epochs, verbose=False)
                history['loss'].extend(chunk_history['loss'])
                
                # Update progress
                progress = (chunk_start + chunk_epochs) / epochs
                progress_bar.progress(progress)
                current_loss = history['loss'][-1]
                status_text.text(f"Epoch {chunk_start + chunk_epochs}/{epochs} - Loss: {current_loss:.6f}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Calculate final accuracy
            predictions = mlp.predict(X_xor)
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = np.mean(predictions_binary == y_xor) * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Training Results")
            
            # Show final metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Final Loss", f"{history['loss'][-1]:.6f}")
            with col_metric2:
                st.metric("Final Accuracy", f"{accuracy:.0f}%")
            with col_metric3:
                convergence_status = "✅ Converged" if accuracy == 100 else "⚠️ Not Fully Converged"
                st.metric("Status", convergence_status)
            
            # Success message
            if accuracy == 100:
                st.success("🎉 **Perfect!** The network learned to solve XOR with 100% accuracy!")
            elif accuracy >= 75:
                st.warning("⚠️ **Partial Success:** The network learned some patterns but got stuck in a local minimum. Try a different seed or increase the learning rate.")
            else:
                st.error("❌ **Training Failed:** The network didn't learn effectively. Try increasing the learning rate or using a different architecture.")
            
            # Plot loss curve
            st.subheader("📉 Loss Curve Over Time")
            fig_loss = plot_loss_curve(history['loss'])
            st.pyplot(fig_loss)
            plt.close(fig_loss)
            
            # Plot decision boundary
            st.subheader("🎯 Decision Boundary")
            fig_boundary = plot_decision_boundary(
                mlp, X_xor, y_xor,
                title=f"Decision Boundary (Accuracy: {accuracy:.0f}%)"
            )
            st.pyplot(fig_boundary)
            plt.close(fig_boundary)
            
            # Show predictions
            st.subheader("🔍 Predictions on XOR Test Cases")
            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
            
            for idx, (col, x_val, y_true, y_pred) in enumerate(zip(
                [pred_col1, pred_col2, pred_col3, pred_col4],
                X_xor, y_xor.flatten(), predictions.flatten()
            )):
                with col:
                    correct = "✅" if (y_pred > 0.5) == y_true else "❌"
                    st.markdown(f"""
                    **Input:** [{x_val[0]}, {x_val[1]}]  
                    **Expected:** {int(y_true)}  
                    **Predicted:** {y_pred:.3f}  
                    **Result:** {correct}
                    """)
    
    # Additional information section
    st.markdown("---")
    st.subheader("🔬 Want to Explore More?")
    
    st.markdown("""
    **Deep Dive into Hyperparameters:**
    - Check out `HYPERPARAMETER_INSIGHTS.md` for detailed explanations of learning rates, architectures, and seed sensitivity
    - Run `explore_hyperparameters.py` to see systematic experiments comparing different configurations
    
    **Key Lessons:**
    1. **Initialization matters:** Some random seeds lead to better starting points than others
    2. **Learning rate is critical:** Too high causes instability, too low causes slow convergence
    3. **Architecture provides robustness:** More neurons = more paths to the solution
    4. **Watch the loss curve:** It tells you everything about training progress
    """)
    
    st.info("""
    💡 **Try This:** Train with seed 42 using 2-2-1 architecture, then try again with 2-4-1. 
    Notice how the larger network is more robust to poor initialization!
    """)


def gradient_flow_tab():
    """Tab 2: Gradient Flow Visualization."""
    st.header("🔄 Gradient Flow Through the Network")
    
    st.markdown("""
    **Understanding Backpropagation:** See how gradients flow backward through the network, 
    computing how much each weight contributed to the error.
    
    This visualization shows the **chain rule** in action - how the error at the output 
    propagates backward through each layer to compute gradients for all weights.
    """)
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Create columns for controls and explanation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Select Test Case")
        
        # Input selection
        test_case_options = {
            "[0, 0] → 0": 0,
            "[0, 1] → 1": 1,
            "[1, 0] → 1": 2,
            "[1, 1] → 0": 3
        }
        test_case_choice = st.selectbox(
            "XOR Input",
            options=list(test_case_options.keys()),
            help="Choose which XOR input to analyze"
        )
        test_case_idx = test_case_options[test_case_choice]
        
        # Get the selected input and target
        X_selected = X_xor[test_case_idx:test_case_idx+1]
        y_selected = y_xor[test_case_idx:test_case_idx+1]
        
        st.markdown(f"""
        **Selected Input:** `{X_selected[0]}`  
        **Expected Output:** `{y_selected[0][0]}`
        """)
        
        # Network configuration
        st.subheader("🏗️ Network Setup")
        architecture = st.selectbox(
            "Architecture",
            options=["2-2-1 (Simple)", "2-4-1 (Robust)"],
            index=0,
            help="Network architecture for gradient visualization"
        )
        layer_sizes = [2, 2, 1] if "2-2-1" in architecture else [2, 4, 1]
        
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Step size for weight updates"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=999,
            value=123,
            help="Seed for weight initialization"
        )
        
        compute_button = st.button("🔍 Compute Gradients", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📚 The Chain Rule")
        
        st.info("""
        **Backpropagation = Chain Rule from Calculus**
        
        The chain rule tells us how to compute derivatives of composed functions:
        
        If `y = f(g(x))`, then `dy/dx = (dy/dg) × (dg/dx)`
        
        In neural networks:
        - **Forward pass:** Compute output layer by layer
        - **Backward pass:** Compute gradients layer by layer (in reverse)
        - **Chain rule:** Multiply gradients as we go backward
        """)
        
        st.success("""
        **Gradient Flow Direction**
        
        1. **Forward →** Input flows through layers to output
        2. **Compute Loss** at the output
        3. **Backward ←** Error flows back through layers
        4. **Update Weights** using computed gradients
        """)
    
    # Compute and visualize gradients
    if compute_button:
        with st.spinner("🔄 Computing forward and backward passes..."):
            # Create network
            mlp = TrainableMLP(
                layer_sizes=layer_sizes,
                activations=['sigmoid'] * (len(layer_sizes) - 1),
                learning_rate=learning_rate,
                random_state=int(random_seed)
            )
            
            # Forward pass with cache
            y_pred, layer_activations, pre_activations = mlp._forward_with_cache(X_selected)
            
            # Compute loss
            loss = mlp._compute_loss(y_selected, y_pred)
            
            # Backward pass
            weight_gradients, bias_gradients = mlp._backward(
                X_selected, y_selected, layer_activations, pre_activations
            )
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Forward Pass Results")
            
            col_fwd1, col_fwd2, col_fwd3 = st.columns(3)
            with col_fwd1:
                st.metric("Input", f"{X_selected[0]}")
            with col_fwd2:
                st.metric("Predicted Output", f"{y_pred[0][0]:.4f}")
            with col_fwd3:
                st.metric("Loss (MSE)", f"{loss:.6f}")
            
            # Show layer-by-layer forward pass
            st.subheader("🔢 Step-by-Step Forward Pass")
            
            for i in range(len(layer_sizes)):
                with st.expander(f"Layer {i}: {layer_sizes[i]} neurons", expanded=(i==0)):
                    if i == 0:
                        st.markdown(f"**Input Layer:** `{layer_activations[i][0]}`")
                    else:
                        st.markdown(f"**Pre-activation (z{i}):** `{pre_activations[i-1][0]}`")
                        st.markdown(f"**After activation:** `{layer_activations[i][0]}`")
                        st.markdown(f"**Activation function:** `{mlp.activations[i-1]}`")
            
            # Show backward pass
            st.markdown("---")
            st.subheader("⬅️ Step-by-Step Backward Pass")
            
            st.markdown("""
            **The Chain Rule in Action:** Starting from the output error, we compute gradients 
            for each layer by multiplying the error by the activation derivative.
            """)
            
            # Output layer gradient
            output_error = y_pred - y_selected
            st.markdown(f"""
            **Step 1: Output Layer Error**
            - Error = Predicted - Target = `{y_pred[0][0]:.4f}` - `{y_selected[0][0]}` = `{output_error[0][0]:.4f}`
            - This error tells us: {"Network output too high" if output_error[0][0] > 0 else "Network output too low"}
            """)
            
            # Show gradients for each layer
            for i in range(len(weight_gradients) - 1, -1, -1):
                with st.expander(f"Layer {i+1} → Layer {i} Gradients", expanded=(i==len(weight_gradients)-1)):
                    st.markdown(f"**Weight Gradients (∂L/∂W{i+1}):**")
                    st.code(np.array2string(weight_gradients[i], precision=4, suppress_small=True))
                    
                    st.markdown(f"**Bias Gradients (∂L/∂b{i+1}):**")
                    st.code(np.array2string(bias_gradients[i], precision=4, suppress_small=True))
                    
                    # Explain gradient magnitude
                    max_weight_grad = np.max(np.abs(weight_gradients[i]))
                    st.markdown(f"**Max gradient magnitude:** `{max_weight_grad:.4f}`")
                    
                    if max_weight_grad > 1.0:
                        st.warning("⚠️ Large gradients - may cause instability")
                    elif max_weight_grad < 0.001:
                        st.info("ℹ️ Small gradients - slow learning")
                    else:
                        st.success("✅ Gradients in good range")
            
            # Show gradient magnitudes visualization
            st.markdown("---")
            st.subheader("📊 Gradient Magnitudes by Layer")
            
            # Create bar chart of gradient magnitudes
            fig_grad_mag, ax = plt.subplots(figsize=(10, 6))
            
            layer_names = [f"Layer {i+1}→{i}" for i in range(len(weight_gradients))]
            avg_gradients = [np.mean(np.abs(wg)) for wg in weight_gradients]
            max_gradients = [np.max(np.abs(wg)) for wg in weight_gradients]
            
            x = np.arange(len(layer_names))
            width = 0.35
            
            ax.bar(x - width/2, avg_gradients, width, label='Average |Gradient|', color='#2E86AB')
            ax.bar(x + width/2, max_gradients, width, label='Max |Gradient|', color='#A23B72')
            
            ax.set_xlabel('Layer Connection', fontsize=12, weight='bold')
            ax.set_ylabel('Gradient Magnitude', fontsize=12, weight='bold')
            ax.set_title('Gradient Magnitudes Across Layers', fontsize=14, weight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig_grad_mag)
            plt.close(fig_grad_mag)
            
            # Show weight updates
            st.markdown("---")
            st.subheader("🔄 Weight Updates (Before → After)")
            
            st.markdown(f"""
            **Gradient Descent Formula:** `W_new = W_old - learning_rate × gradient`
            
            With learning rate = `{learning_rate}`, we update each weight by moving in the 
            **opposite direction** of the gradient (downhill).
            """)
            
            for i in range(len(weight_gradients)):
                with st.expander(f"Layer {i+1} Weight Updates", expanded=(i==0)):
                    # Compute updated weights
                    old_weights = mlp.weights_[i].copy()
                    new_weights = old_weights - learning_rate * weight_gradients[i]
                    weight_change = new_weights - old_weights
                    
                    col_w1, col_w2, col_w3 = st.columns(3)
                    
                    with col_w1:
                        st.markdown("**Before:**")
                        st.code(np.array2string(old_weights, precision=3, suppress_small=True))
                    
                    with col_w2:
                        st.markdown("**Change (Δ):**")
                        st.code(np.array2string(weight_change, precision=3, suppress_small=True))
                    
                    with col_w3:
                        st.markdown("**After:**")
                        st.code(np.array2string(new_weights, precision=3, suppress_small=True))
                    
                    # Show bias updates
                    old_biases = mlp.biases_[i].copy()
                    new_biases = old_biases - learning_rate * bias_gradients[i]
                    bias_change = new_biases - old_biases
                    
                    st.markdown("**Bias Updates:**")
                    st.markdown(f"- Before: `{old_biases}`")
                    st.markdown(f"- Change: `{bias_change}`")
                    st.markdown(f"- After: `{new_biases}`")
            
            # Network diagram with gradients
            st.markdown("---")
            st.subheader("🕸️ Network Diagram with Gradients")
            
            st.markdown("""
            **Visualization Key:**
            - **Node colors:** Green=Input, Blue=Hidden, Red=Output
            - **Connection colors:** Red=Positive weight, Blue=Negative weight
            - **Connection thickness:** Proportional to weight magnitude
            - **Connection opacity:** Proportional to gradient magnitude (darker = larger gradient)
            - **Labels:** Show weight values and gradients (∇)
            """)
            
            fig_network = plot_network_with_gradients(
                mlp, weight_gradients, bias_gradients
            )
            st.pyplot(fig_network)
            plt.close(fig_network)
            
            # Key insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.info("""
                **Chain Rule Application**
                
                Each layer's gradient depends on:
                1. The gradient from the next layer (flowing backward)
                2. The activation derivative at this layer
                3. The input to this layer
                
                This is the chain rule: multiply derivatives as we go backward!
                """)
            
            with col_insight2:
                st.success("""
                **Gradient Descent Update**
                
                Weights move in the **opposite direction** of gradients:
                - Positive gradient → Decrease weight
                - Negative gradient → Increase weight
                - Large gradient → Large update
                - Small gradient → Small update
                
                This minimizes the loss function!
                """)
            
            # Try this section
            st.markdown("---")
            st.info("""
            💡 **Try This:**
            1. Compare gradients for different test cases - notice how they differ based on the error
            2. Try different learning rates - see how it affects the weight update magnitude
            3. Look at the network diagram - connections with larger gradients need more adjustment
            4. Watch how gradients get smaller in earlier layers (vanishing gradient effect with sigmoid)
            """)

if __name__ == "__main__":
    main()
