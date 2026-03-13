"""
Streamlit interactive playground for experimenting with regularization.

This app demonstrates overfitting and how dropout and weight decay prevent it.
Users can adjust regularization parameters and see the effects in real-time.

Features:
- Tab 1: Overfitting Demonstration - Train networks with different regularization settings
- Tab 2: Weight Distribution - Compare weight distributions across configurations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from network_with_regularization import RegularizedNetwork, RegularizedTrainer
from overfitting_demo import Adam, load_mnist_data


# Configure page layout
st.set_page_config(
    page_title="Regularization Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Regularization Playground: Preventing Overfitting")
st.markdown("""
Explore how **dropout** and **weight decay** prevent neural networks from overfitting.
Adjust the regularization parameters and watch the train vs test accuracy gap change in real-time.
""")

# Load MNIST data (cached to avoid reloading)
@st.cache_data
def get_mnist_data():
    """Load and cache MNIST data for the playground."""
    return load_mnist_data(num_train=50000, num_test=10000)

# Load data
(X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = get_mnist_data()

# Check if we're using real MNIST data
try:
    import tensorflow
    using_real_data = True
except ImportError:
    using_real_data = False
    st.error("""
    ⚠️ **TensorFlow not installed - using synthetic random data!**
    
    The playground requires real MNIST data to work properly. Please install TensorFlow:
    ```bash
    pip install tensorflow
    ```
    
    The current synthetic data is random noise, so the network won't learn properly.
    Accuracy will remain very low (~10%) regardless of settings.
    """)

# Create tabs
tab1, tab2 = st.tabs(["📊 Overfitting Demonstration", "📈 Weight Distribution"])

# ============================================================================
# TAB 1: OVERFITTING DEMONSTRATION
# ============================================================================

with tab1:
    st.header("Watch Overfitting Happen (and How to Stop It)")
    
    st.markdown("""
    **How to use this tab:**
    1. Adjust the regularization parameters on the left
    2. Click "Train Network" to start training
    3. Watch the accuracy curves update in real-time
    4. Notice how the overfitting gap (orange area) changes with regularization
    """)
    
    # Create two columns: controls on left, visualizations on right
    col1, col2 = st.columns([1, 2.5])
    
    # ========================================================================
    # SUBTASK 9.2: UI CONTROLS
    # ========================================================================
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        # Network size dropdown
        st.write("**Network Architecture:**")
        hidden_size = st.selectbox(
            "Network Size",
            [128, 256, 512],
            index=1,
            help="Number of neurons in the hidden layer"
        )
        st.write(f"Architecture: 784 → {hidden_size} → 10")
        
        # Regularization controls
        st.write("**Regularization Settings:**")
        
        dropout_rate = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.9,
            value=0.0,
            step=0.1,
            help="Probability of deactivating each neuron during training (0.0 = no dropout)"
        )
        
        weight_decay = st.slider(
            "Weight Decay",
            min_value=0.0,
            max_value=0.01,
            value=0.0,
            step=0.001,
            format="%.3f",
            help="L2 regularization coefficient (0.001 = standard, 0.01 = aggressive)"
        )
        
        # Training controls
        st.write("**Training Settings:**")
        
        epochs = st.number_input(
            "Epochs",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="Number of training epochs"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [32, 64, 128],
            index=1,
            help="Mini-batch size for training"
        )
        
        # Train button
        train_button = st.button("🚀 Train Network", type="primary", use_container_width=True)
        
        # Display current settings summary
        st.divider()
        st.write("**Current Settings Summary:**")
        st.write(f"- Dropout: {dropout_rate:.1%}")
        st.write(f"- Weight Decay: {weight_decay:.4f}")
        st.write(f"- Network: 784 → {hidden_size} → 10")
        st.write(f"- Epochs: {epochs}")
        st.write(f"- Batch Size: {batch_size}")
    
    # ========================================================================
    # SUBTASK 9.3 & 9.4: TRAINING EXECUTION AND VISUALIZATION
    # ========================================================================
    
    with col2:
        if train_button:
            # Create placeholders for status and progress
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            status_placeholder.info("🔄 Initializing network...")
            progress_bar = progress_placeholder.progress(0)
            
            try:
                # Create and train network
                status_placeholder.info("🔄 Training network...")
                
                network = RegularizedNetwork(
                    input_size=784,
                    hidden_size=hidden_size,
                    output_size=10,
                    dropout_rate=dropout_rate,
                    seed=42
                )
                
                optimizer = Adam(learning_rate=0.001)
                trainer = RegularizedTrainer(network, optimizer, weight_decay=weight_decay)
                
                start_time = time.time()
                
                # Train with progress updates
                history = trainer.train(
                    X_train, y_train_onehot, y_train,
                    X_test, y_test_onehot, y_test,
                    epochs=int(epochs),
                    batch_size=batch_size,
                    verbose=False
                )
                
                training_time = time.time() - start_time
                
                # Update progress bar
                progress_bar.progress(1.0)
                status_placeholder.success("✅ Training complete!")
                
                # ============================================================
                # SUBTASK 9.3: VISUALIZATIONS
                # ============================================================
                
                st.subheader("📊 Training Results")
                
                # Create 2x2 grid of plots
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle("Training Metrics", fontsize=16, fontweight='bold')
                
                # Plot 1: Accuracy curves (train vs test)
                ax = axes[0, 0]
                epochs_range = range(len(history['train_acc']))
                ax.plot(epochs_range, history['train_acc'], label='Train Accuracy', 
                       linewidth=2.5, color='#1f77b4', marker='o', markersize=4)
                ax.plot(epochs_range, history['test_acc'], label='Test Accuracy', 
                       linewidth=2.5, color='#ff7f0e', marker='s', markersize=4)
                ax.fill_between(epochs_range, history['train_acc'], history['test_acc'], 
                               alpha=0.2, color='#d62728', label='Overfitting Gap')
                ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
                ax.set_ylabel("Accuracy", fontsize=11, fontweight='bold')
                ax.set_title("Train vs Test Accuracy", fontsize=12, fontweight='bold')
                ax.legend(fontsize=10, loc='lower right')
                ax.grid(True, alpha=0.3, linestyle='--')
                # Dynamic y-axis based on actual accuracy values
                min_acc = min(min(history['train_acc']), min(history['test_acc']))
                max_acc = max(max(history['train_acc']), max(history['test_acc']))
                margin = (max_acc - min_acc) * 0.1 if (max_acc - min_acc) > 0 else 0.1
                ax.set_ylim([max(0, min_acc - margin), min(1.0, max_acc + margin)])
                
                # Plot 2: Loss curves (train vs test)
                ax = axes[0, 1]
                ax.plot(epochs_range, history['train_loss'], label='Train Loss', 
                       linewidth=2.5, color='#2ca02c', marker='o', markersize=4)
                ax.plot(epochs_range, history['test_loss'], label='Test Loss', 
                       linewidth=2.5, color='#d62728', marker='s', markersize=4)
                ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
                ax.set_ylabel("Loss", fontsize=11, fontweight='bold')
                ax.set_title("Train vs Test Loss", fontsize=12, fontweight='bold')
                ax.legend(fontsize=10, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Plot 3: Loss decomposition (CE loss vs penalty)
                ax = axes[1, 0]
                ax.plot(epochs_range, history['ce_loss'], label='Cross-Entropy Loss', 
                       linewidth=2.5, color='#9467bd', marker='o', markersize=4)
                ax.plot(epochs_range, history['penalty'], label='Weight Decay Penalty', 
                       linewidth=2.5, color='#ff9896', marker='s', markersize=4)
                ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
                ax.set_ylabel("Loss Component", fontsize=11, fontweight='bold')
                ax.set_title("Loss Decomposition", fontsize=12, fontweight='bold')
                ax.legend(fontsize=10, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Plot 4: Overfitting gap
                ax = axes[1, 1]
                gap = np.array(history['train_acc']) - np.array(history['test_acc'])
                ax.plot(epochs_range, gap, linewidth=2.5, color='#d62728', marker='o', markersize=4)
                ax.fill_between(epochs_range, gap, alpha=0.3, color='#d62728')
                ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
                ax.set_ylabel("Accuracy Gap", fontsize=11, fontweight='bold')
                ax.set_title("Overfitting Gap (Train - Test)", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_ylim([0, max(gap) * 1.15 if len(gap) > 0 else 0.1])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # ============================================================
                # SUBTASK 9.4: RESULTS DISPLAY
                # ============================================================
                
                st.subheader("📈 Performance Summary")
                
                # Calculate final metrics
                final_train_acc = history['train_acc'][-1]
                final_test_acc = history['test_acc'][-1]
                final_gap = final_train_acc - final_test_acc
                
                # Display metrics in columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Final Train Accuracy",
                        f"{final_train_acc:.4f}",
                        delta=f"{final_train_acc - history['train_acc'][0]:.4f}",
                        delta_color="normal"
                    )
                
                with metric_col2:
                    st.metric(
                        "Final Test Accuracy",
                        f"{final_test_acc:.4f}",
                        delta=f"{final_test_acc - history['test_acc'][0]:.4f}",
                        delta_color="normal"
                    )
                
                with metric_col3:
                    st.metric(
                        "Overfitting Gap",
                        f"{final_gap:.4f}",
                        delta=f"{final_gap - (history['train_acc'][0] - history['test_acc'][0]):.4f}",
                        delta_color="inverse"
                    )
                
                with metric_col4:
                    st.metric(
                        "Training Time",
                        f"{training_time:.2f}s"
                    )
                
                # Interpretation section
                st.subheader("💡 Interpretation")
                
                if final_test_acc < 0.2:
                    st.error(
                        "❌ **Network is not learning!** Test accuracy is extremely low. "
                        "This usually means regularization is too strong. Try reducing weight decay or dropout rate."
                    )
                elif final_gap < 0.02:
                    st.success(
                        "✅ **Excellent Generalization!** Train and test accuracy are very close. "
                        "The network is learning generalizable patterns, not memorizing the training data."
                    )
                elif final_gap < 0.05:
                    st.info(
                        "ℹ️ **Good Generalization.** Some overfitting is present but acceptable. "
                        "The regularization is helping, but you might benefit from stronger regularization."
                    )
                else:
                    st.warning(
                        "⚠️ **Significant Overfitting Detected.** The gap between train and test accuracy is large. "
                        "Try increasing dropout rate or weight decay to improve generalization."
                    )
                
                # Additional insights
                st.markdown("---")
                st.write("**Key Observations:**")
                
                insights = []
                
                if dropout_rate > 0:
                    insights.append(f"• **Dropout ({dropout_rate:.1%})** is randomly deactivating neurons during training, "
                                  "preventing co-adaptation and improving generalization.")
                
                if weight_decay > 0:
                    insights.append(f"• **Weight Decay ({weight_decay:.4f})** is penalizing large weights, "
                                  "encouraging simpler models that generalize better.")
                
                if dropout_rate == 0 and weight_decay == 0:
                    insights.append("• **No regularization** is applied. The network may overfit on the training data. "
                                  "Try enabling dropout or weight decay to see the difference!")
                
                if final_gap > 0.1:
                    insights.append(f"• The overfitting gap ({final_gap:.4f}) is quite large. "
                                  "Consider increasing regularization strength.")
                
                if final_train_acc > 0.99:
                    insights.append("• Training accuracy is very high, which is good, but make sure test accuracy is also high!")
                
                for insight in insights:
                    st.write(insight)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                progress_placeholder.empty()
                status_placeholder.empty()
        
        else:
            # Show placeholder when no training has been run
            st.info(
                "👈 **Configure the network on the left and click 'Train Network' to start!**\n\n"
                "Try these experiments:\n"
                "1. Train with **no regularization** to see overfitting\n"
                "2. Add **dropout** and watch the gap shrink\n"
                "3. Add **weight decay** for additional improvement\n"
                "4. Combine both for best generalization"
            )


# ============================================================================
# TAB 2: WEIGHT DISTRIBUTION
# ============================================================================

with tab2:
    st.header("Weight Distribution Analysis")
    st.markdown("""
    Compare weight distributions with and without regularization.
    Notice how **weight decay** keeps weights smaller and more concentrated near zero,
    while **dropout** doesn't directly affect weight magnitudes but improves generalization.
    """)
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        st.write("**Regularization Techniques:**")
        compare_dropout = st.checkbox("Include Dropout", value=True)
        compare_weight_decay = st.checkbox("Include Weight Decay", value=True)
        
        train_button_2 = st.button("🚀 Train and Compare", type="primary", use_container_width=True)
    
    with col2:
        if train_button_2:
            # Define configurations to compare
            configs = [
                ("No Regularization", 0.0, 0.0),
            ]
            
            if compare_dropout:
                configs.append(("Dropout Only (0.3)", 0.3, 0.0))
            
            if compare_weight_decay:
                configs.append(("Weight Decay Only (0.001)", 0.0, 0.001))
            
            if compare_dropout and compare_weight_decay:
                configs.append(("Both (Dropout 0.3 + WD 0.001)", 0.3, 0.001))
            
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            status_placeholder.info("🔄 Training networks...")
            progress_bar = progress_placeholder.progress(0)
            
            try:
                # Train multiple networks
                trained_networks = {}
                
                for idx, (name, dropout, decay) in enumerate(configs):
                    status_placeholder.info(f"🔄 Training: {name}...")
                    
                    network = RegularizedNetwork(
                        input_size=784,
                        hidden_size=256,
                        output_size=10,
                        dropout_rate=dropout,
                        seed=42
                    )
                    
                    optimizer = Adam(learning_rate=0.001)
                    trainer = RegularizedTrainer(network, optimizer, weight_decay=decay)
                    
                    trainer.train(
                        X_train, y_train_onehot, y_train,
                        X_test, y_test_onehot, y_test,
                        epochs=10,
                        batch_size=64,
                        verbose=False
                    )
                    
                    trained_networks[name] = network
                    progress_bar.progress((idx + 1) / len(configs))
                
                status_placeholder.success("✅ Training complete!")
                progress_placeholder.empty()
                
                # Create weight distribution plots
                st.subheader("📊 Weight Distribution Comparison")
                
                fig, axes = plt.subplots(1, len(configs), figsize=(5*len(configs), 4))
                if len(configs) == 1:
                    axes = [axes]
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                for idx, (name, _, _) in enumerate(configs):
                    ax = axes[idx]
                    network = trained_networks[name]
                    
                    # Get weights from first layer
                    weights = network.params['W1'].flatten()
                    
                    # Plot histogram
                    ax.hist(weights, bins=50, alpha=0.7, edgecolor='black', color=colors[idx])
                    
                    # Add statistics
                    mean_weight = np.mean(weights)
                    std_weight = np.std(weights)
                    
                    ax.set_title(f"{name}\nMean: {mean_weight:.4f} | Std: {std_weight:.4f}",
                               fontsize=11, fontweight='bold')
                    ax.set_xlabel("Weight Value", fontsize=10)
                    ax.set_ylabel("Frequency", fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Add vertical line at mean
                    ax.axvline(mean_weight, color='red', linestyle='--', linewidth=2, label='Mean')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Insights
                st.markdown("---")
                st.write("**Key Observations:**")
                
                st.write(
                    "• **Weight Decay** keeps weights smaller and more concentrated near zero, "
                    "which encourages simpler models that generalize better."
                )
                st.write(
                    "• **Dropout** doesn't directly affect weight magnitudes, but it prevents "
                    "co-adaptation by randomly deactivating neurons during training."
                )
                st.write(
                    "• **Combined regularization** (dropout + weight decay) provides the best "
                    "balance between model capacity and generalization."
                )
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                progress_placeholder.empty()
                status_placeholder.empty()
        
        else:
            st.info(
                "👈 **Select regularization techniques and click 'Train and Compare' to start!**\n\n"
                "This tab trains multiple networks with different regularization settings "
                "and compares their weight distributions."
            )


# Footer
st.markdown("---")
st.markdown(
    """
    **About this playground:** This interactive tool demonstrates overfitting and regularization
    techniques (dropout and weight decay) on the MNIST dataset. All computations are done from scratch
    using NumPy, without relying on deep learning frameworks.
    
    **Blog Post:** This playground is part of the "Perceptrons to Transformers" blog series.
    See the accompanying blog post for detailed explanations of overfitting, dropout, and weight decay.
    """
)
