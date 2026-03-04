# Streamlit interactive playground for blog post #4

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from mnist_trainer import load_mnist, Network, Trainer
from optimizers import SGD, MomentumSGD, Adam
import time
from io import BytesIO

# Configure page layout and title
st.set_page_config(page_title="Optimization Playground", layout="wide")
st.title("🎯 Optimization Playground: From SGD to Adam")

# Load MNIST data (cached for performance)
@st.cache_data
def get_mnist_data():
    """Load and cache MNIST dataset."""
    return load_mnist()

(X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = get_mnist_data()

# Main content: Optimizer Comparison
st.header("Compare Optimizers on MNIST")
st.write("Train neural networks with different optimizers and compare their performance.")

# Create two columns: controls on left, results on right
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Training Configuration")
    
    # Optimizer selection checkboxes
    st.write("**Select Optimizers:**")
    use_sgd = st.checkbox("SGD", value=True)
    use_momentum = st.checkbox("Momentum", value=True)
    use_adam = st.checkbox("Adam", value=True)
    
    # Hyperparameters
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
    
    # Train button
    train_button = st.button("🚀 Train Networks", type="primary")

with col2:
    # Training execution and visualization
    if train_button:
        # Collect selected optimizers
        selected_optimizers = []
        if use_sgd:
            selected_optimizers.append(("SGD", SGD(learning_rate)))
        if use_momentum:
            selected_optimizers.append(("Momentum", MomentumSGD(learning_rate)))
        if use_adam:
            selected_optimizers.append(("Adam", Adam(learning_rate)))
        
        if not selected_optimizers:
            st.warning("⚠️ Please select at least one optimizer!")
        else:
            results = {}
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train each optimizer
            for idx, (name, optimizer) in enumerate(selected_optimizers):
                status_text.text(f"Training {name}... ({idx+1}/{len(selected_optimizers)})")
                
                # Create fresh network for each optimizer
                network = Network(seed=42)
                trainer = Trainer(network, optimizer)
                
                # Track training time
                start_time = time.time()
                history = trainer.train(
                    X_train, y_train_onehot, y_train,
                    X_test, y_test_onehot, y_test,
                    epochs=int(epochs), batch_size=batch_size, verbose=False
                )
                training_time = time.time() - start_time
                
                # Store results
                results[name] = {
                    'history': history,
                    'time': training_time,
                    'network': network
                }
                
                # Update progress
                progress_bar.progress((idx + 1) / len(selected_optimizers))
            
            status_text.text("✅ Training complete!")
            
            # Visualize results
            st.subheader("Training Accuracy Comparison")
            
            # Define distinct colors and markers for each optimizer
            optimizer_styles = {
                'Adam': {'color': '#d62728', 'marker': 'D'},          # Red, diamond
                'Momentum': {'color': '#2ca02c', 'marker': '^'},     # Green, triangle
                'SGD': {'color': '#ff7f0e', 'marker': 's'}          # Orange, square
            }
            
            # Create single accuracy plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot training accuracy curves for all optimizers (in alphabetical order for consistency)
            optimizer_names = sorted(results.keys())
            
            for idx, opt_name in enumerate(optimizer_names):
                result = results[opt_name]
                history = result['history']
                epochs_list = list(range(1, len(history['train_acc']) + 1))
                
                style = optimizer_styles.get(opt_name, {'color': '#999999', 'marker': 'x'})
                
                # Convert accuracy to percentage (0-100 scale)
                train_acc_percent = [acc * 100 for acc in history['train_acc']]
                
                # Plot training accuracy with clear styling
                ax.plot(
                    epochs_list,
                    train_acc_percent,
                    label=opt_name,
                    color=style['color'],
                    marker=style['marker'],
                    markersize=10,
                    linewidth=3,
                    linestyle='-',
                    alpha=0.9,
                    markeredgewidth=2,
                    markeredgecolor='white',
                    zorder=10 + idx,
                    clip_on=False
                )
            
            # Configure plot
            ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
            ax.set_ylabel("Training Accuracy (%)", fontsize=14, fontweight='bold')
            ax.set_title("Optimizer Comparison: Training Accuracy", fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='gray')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_xlim(left=0.5)
            ax.set_ylim(bottom=0, top=105)
            
            # Finalize plot
            plt.tight_layout()
            
            # Save to buffer and display as image
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, width='stretch')
            plt.close(fig)
            
            st.caption("📊 Training accuracy over epochs for each optimizer. Higher is better!")
            
            # Performance summary table
            st.subheader("Performance Summary")
            summary_data = []
            for name, result in results.items():
                final_acc = result['history']['train_acc'][-1] * 100  # Convert to percentage
                summary_data.append({
                    "Optimizer": name,
                    "Final Training Accuracy": f"{final_acc:.2f}%",
                    "Training Time (s)": f"{result['time']:.2f}"
                })
            st.table(summary_data)
            
            # Sample predictions visualization
            st.subheader("Sample Predictions")
            st.write("Showing 10 random test images with predictions from the first trained optimizer.")
            
            # Select 10 random test samples
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(len(X_test), 10, replace=False)
            
            # Get the first optimizer's network for predictions
            first_optimizer_name = list(results.keys())[0]
            network = results[first_optimizer_name]['network']
            
            # Create visualization
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for idx, sample_idx in enumerate(sample_indices):
                ax = axes[idx // 5, idx % 5]
                
                # Display image
                image = X_test[sample_idx].reshape(28, 28)
                ax.imshow(image, cmap='gray')
                ax.axis('off')
                
                # Get prediction
                pred = network.predict(X_test[sample_idx:sample_idx+1])[0]
                true_label = y_test[sample_idx]
                
                # Color-code: green for correct, red for incorrect
                color = 'green' if pred == true_label else 'red'
                ax.set_title(f"Pred: {pred}\nTrue: {true_label}", 
                            color=color, fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            # Save to buffer and display as image
            buf2 = BytesIO()
            fig.savefig(buf2, format='png', dpi=120, bbox_inches='tight')
            buf2.seek(0)
            st.image(buf2, width='stretch')
            plt.close(fig)
            
            # Store results in session state for potential reuse
            st.session_state['training_results'] = results
    else:
        st.info("Configure settings and click 'Train Networks' to begin.")
