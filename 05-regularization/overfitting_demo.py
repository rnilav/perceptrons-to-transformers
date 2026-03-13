"""
Script to generate overfitting examples and visualizations.

This script demonstrates:
1. How networks overfit without regularization (train acc → 100%, test acc plateaus)
2. How dropout and weight decay reduce overfitting
3. Comparison of different regularization combinations

Generates PNG visualizations for use in the blog post.
"""

import numpy as np
import os
from network_with_regularization import RegularizedNetwork, RegularizedTrainer


# Simple optimizer for training (reuse from post #4 or implement minimal version)
class Adam:
    """Adam optimizer for neural network training."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (default 0.001)
            beta1: Exponential decay rate for first moment (default 0.9)
            beta2: Exponential decay rate for second moment (default 0.999)
            epsilon: Small constant for numerical stability (default 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def update(self, params: dict, gradients: dict) -> dict:
        """Update parameters using Adam algorithm.
        
        Args:
            params: Dictionary of current parameters
            gradients: Dictionary of gradients
            
        Returns:
            Updated parameters
        """
        self.t += 1
        updated_params = params.copy()
        
        for key in params:
            # Initialize moment estimates if needed
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


def load_mnist_data(num_train: int = 50000, num_test: int = 10000):
    """Load and preprocess MNIST data.
    
    Args:
        num_train: Number of training samples to use (default 50000)
        num_test: Number of test samples to use (default 10000)
        
    Returns:
        Tuple of ((X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot))
    """
    # Try TensorFlow first
    try:
        from tensorflow.keras.datasets import mnist
        print("Loading MNIST from TensorFlow...")
        (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
        
        # Flatten and normalize
        X_train_full = X_train_full.reshape(-1, 784) / 255.0
        X_test_full = X_test_full.reshape(-1, 784) / 255.0
        
        # Use subset for faster training
        X_train = X_train_full[:num_train]
        y_train = y_train_full[:num_train]
        X_test = X_test_full[:num_test]
        y_test = y_test_full[:num_test]
        
        print(f"✓ MNIST loaded from TensorFlow: {X_train.shape}")
        
    except (ImportError, ModuleNotFoundError):
        # Try scikit-learn as fallback
        try:
            from sklearn.datasets import load_digits
            print("Loading MNIST from scikit-learn...")
            
            # Load digits dataset (8x8 images, 1797 samples)
            # This is a smaller version of MNIST
            digits = load_digits()
            X_full = digits.data / 16.0  # Normalize to [0, 1]
            y_full = digits.target
            
            # Repeat data to get enough samples
            repeats = max(1, num_train // len(X_full) + 1)
            X_full = np.tile(X_full, (repeats, 1))
            y_full = np.tile(y_full, repeats)
            
            # Add some noise to make it more interesting
            np.random.seed(42)
            X_full = X_full + np.random.normal(0, 0.01, X_full.shape)
            X_full = np.clip(X_full, 0, 1)
            
            # Pad to 784 dimensions (28x28) to match MNIST
            X_full_padded = np.zeros((X_full.shape[0], 784))
            X_full_padded[:, :64] = X_full
            X_full = X_full_padded
            
            # Split into train/test
            X_train = X_full[:num_train]
            y_train = y_full[:num_train]
            X_test = X_full[num_train:num_train+num_test]
            y_test = y_full[num_train:num_train+num_test]
            
            print(f"✓ Digits dataset loaded from scikit-learn: {X_train.shape}")
            
        except (ImportError, ModuleNotFoundError):
            # Final fallback: download MNIST directly
            try:
                import urllib.request
                import gzip
                import pickle
                
                print("Downloading MNIST from online source...")
                
                url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
                filename = "/tmp/mnist.pkl.gz"
                
                urllib.request.urlretrieve(url, filename)
                
                with gzip.open(filename, 'rb') as f:
                    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
                
                X_train_full, y_train_full = train_set
                X_test_full, y_test_full = test_set
                
                X_train = X_train_full[:num_train]
                y_train = y_train_full[:num_train]
                X_test = X_test_full[:num_test]
                y_test = y_test_full[:num_test]
                
                print(f"✓ MNIST downloaded: {X_train.shape}")
                
            except Exception as e:
                print("=" * 70)
                print("WARNING: Cannot load real MNIST data!")
                print("=" * 70)
                print(f"Error: {e}")
                print()
                print("Tried:")
                print("  1. TensorFlow (not available in this environment)")
                print("  2. scikit-learn (not available)")
                print("  3. Direct download (failed)")
                print()
                print("Generating synthetic random data as fallback...")
                print("(This data is random noise - the network won't learn properly)")
                print()
                print("To fix this, install one of:")
                print("  pip install tensorflow")
                print("  pip install scikit-learn")
                print("=" * 70)
                
                # Generate synthetic data as fallback
                np.random.seed(42)
                X_train = np.random.randn(num_train, 784) * 0.5 + 0.5
                X_train = np.clip(X_train, 0, 1)
                y_train = np.random.randint(0, 10, num_train)
                
                X_test = np.random.randn(num_test, 784) * 0.5 + 0.5
                X_test = np.clip(X_test, 0, 1)
                y_test = np.random.randint(0, 10, num_test)
    
    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    
    return (X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot)


def train_network(X_train, y_train_onehot, y_train_labels,
                  X_test, y_test_onehot, y_test_labels,
                  hidden_size: int = 256,
                  dropout_rate: float = 0.0,
                  weight_decay: float = 0.0,
                  epochs: int = 30,
                  batch_size: int = 64,
                  seed: int = 42):
    """Train a regularized network and return history.
    
    Args:
        X_train, y_train_onehot, y_train_labels: Training data
        X_test, y_test_onehot, y_test_labels: Test data
        hidden_size: Number of hidden neurons
        dropout_rate: Dropout rate (0.0 to 0.9)
        weight_decay: Weight decay coefficient
        epochs: Number of training epochs
        batch_size: Mini-batch size
        seed: Random seed for reproducibility
        
    Returns:
        Training history dictionary
    """
    network = RegularizedNetwork(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        dropout_rate=dropout_rate,
        seed=seed
    )
    
    optimizer = Adam(learning_rate=0.001)
    trainer = RegularizedTrainer(network, optimizer, weight_decay=weight_decay)
    
    history = trainer.train(
        X_train, y_train_onehot, y_train_labels,
        X_test, y_test_onehot, y_test_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False
    )
    
    return history


def plot_overfitting_example(history, title: str, filename: str):
    """Create a single overfitting visualization.
    
    Args:
        history: Training history dictionary
        title: Title for the plot
        filename: Filename to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='blue')
    ax.plot(history['test_acc'], label='Test Accuracy', linewidth=2, color='red')
    ax.fill_between(range(len(history['train_acc'])), 
                    history['train_acc'], history['test_acc'], 
                    alpha=0.2, color='orange', label='Overfitting Gap')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Train vs Test Accuracy", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.02])
    
    # Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='green')
    ax.plot(history['test_loss'], label='Test Loss', linewidth=2, color='red')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Train vs Test Loss", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Loss components
    ax = axes[1, 0]
    ax.plot(history['ce_loss'], label='Cross-Entropy Loss', linewidth=2, color='purple')
    ax.plot(history['penalty'], label='Weight Decay Penalty', linewidth=2, color='orange')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss Component", fontsize=11)
    ax.set_title("Loss Decomposition", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax = axes[1, 1]
    gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    ax.plot(gap, linewidth=2, color='orange')
    ax.fill_between(range(len(gap)), gap, alpha=0.3, color='orange')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy Gap", fontsize=11)
    ax.set_title("Overfitting Gap (Train - Test)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(gap) * 1.1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def generate_overfitting_examples():
    """Generate overfitting examples showing train acc → 100%, test acc plateaus.
    
    This is subtask 7.1: Create script to generate overfitting examples
    """
    print("\n" + "="*70)
    print("SUBTASK 7.1: Generating Overfitting Examples")
    print("="*70)
    
    # Load data
    print("Loading MNIST data...")
    (X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = load_mnist_data(
        num_train=5000, num_test=1000
    )
    
    # Train network with NO regularization for many epochs
    print("Training network with NO regularization (30 epochs)...")
    history_no_reg = train_network(
        X_train, y_train_onehot, y_train,
        X_test, y_test_onehot, y_test,
        hidden_size=256,
        dropout_rate=0.0,
        weight_decay=0.0,
        epochs=30,
        batch_size=64,
        seed=42
    )
    
    # Print final metrics
    final_train_acc = history_no_reg['train_acc'][-1]
    final_test_acc = history_no_reg['test_acc'][-1]
    final_gap = final_train_acc - final_test_acc
    
    print(f"\nNo Regularization Results:")
    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    print(f"  Final Test Accuracy:  {final_test_acc:.4f}")
    print(f"  Overfitting Gap:      {final_gap:.4f}")
    
    # Create visualization
    os.makedirs("visualizations", exist_ok=True)
    plot_overfitting_example(
        history_no_reg,
        "Overfitting Without Regularization\n(Train Accuracy → 100%, Test Accuracy Plateaus)",
        "visualizations/01_overfitting_no_regularization.png"
    )
    
    print("\n✓ Subtask 7.1 complete: Overfitting examples generated")


def generate_comparison_visualizations():
    """Generate comparison visualizations with different regularization settings.
    
    This is subtask 7.2: Create comparison script with before/after visualizations
    """
    print("\n" + "="*70)
    print("SUBTASK 7.2: Generating Comparison Visualizations")
    print("="*70)
    
    # Load data
    print("Loading MNIST data...")
    (X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = load_mnist_data(
        num_train=5000, num_test=1000
    )
    
    # Define configurations to compare
    configs = [
        ("No Regularization", 0.0, 0.0),
        ("Dropout Only (0.3)", 0.3, 0.0),
        ("Weight Decay Only (0.001)", 0.0, 0.001),
        ("Both (Dropout 0.3 + WD 0.001)", 0.3, 0.001),
    ]
    
    histories = {}
    
    # Train networks with different regularization settings
    for name, dropout, weight_decay in configs:
        print(f"\nTraining: {name}...")
        history = train_network(
            X_train, y_train_onehot, y_train,
            X_test, y_test_onehot, y_test,
            hidden_size=256,
            dropout_rate=dropout,
            weight_decay=weight_decay,
            epochs=30,
            batch_size=64,
            seed=42
        )
        histories[name] = history
        
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        final_gap = final_train_acc - final_test_acc
        
        print(f"  Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}, Gap: {final_gap:.4f}")
    
    # Create individual visualizations for each configuration
    os.makedirs("visualizations", exist_ok=True)
    
    filenames = [
        "visualizations/02_no_regularization.png",
        "visualizations/03_dropout_only.png",
        "visualizations/04_weight_decay_only.png",
        "visualizations/05_both_regularization.png",
    ]
    
    for (name, _, _), filename in zip(configs, filenames):
        plot_overfitting_example(histories[name], f"Regularization: {name}", filename)
    
    # Create comparison plot: all accuracy curves on one chart
    print("\nCreating comparison plot...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Regularization Comparison: Impact on Overfitting", fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'purple']
    
    # Plot 1: Train Accuracy Comparison
    ax = axes[0, 0]
    for (name, _, _), color in zip(configs, colors):
        ax.plot(histories[name]['train_acc'], label=name, linewidth=2, color=color, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train Accuracy", fontsize=11)
    ax.set_title("Training Accuracy Comparison", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.02])
    
    # Plot 2: Test Accuracy Comparison
    ax = axes[0, 1]
    for (name, _, _), color in zip(configs, colors):
        ax.plot(histories[name]['test_acc'], label=name, linewidth=2, color=color, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Test Accuracy Comparison", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.02])
    
    # Plot 3: Overfitting Gap Comparison
    ax = axes[1, 0]
    for (name, _, _), color in zip(configs, colors):
        gap = np.array(histories[name]['train_acc']) - np.array(histories[name]['test_acc'])
        ax.plot(gap, label=name, linewidth=2, color=color, linestyle='-')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Overfitting Gap", fontsize=11)
    ax.set_title("Overfitting Gap Comparison (Train - Test)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Final Metrics Summary\n" + "="*40 + "\n\n"
    for name, _, _ in configs:
        train_acc = histories[name]['train_acc'][-1]
        test_acc = histories[name]['test_acc'][-1]
        gap = train_acc - test_acc
        summary_text += f"{name}:\n"
        summary_text += f"  Train: {train_acc:.4f}\n"
        summary_text += f"  Test:  {test_acc:.4f}\n"
        summary_text += f"  Gap:   {gap:.4f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("visualizations/06_comparison_all_metrics.png", dpi=150, bbox_inches='tight')
    print("Saved: visualizations/06_comparison_all_metrics.png")
    plt.close()
    
    print("\n✓ Subtask 7.2 complete: Comparison visualizations generated")


def main():
    """Main entry point for the overfitting demonstration script."""
    print("\n" + "="*70)
    print("OVERFITTING DEMONSTRATION SCRIPT")
    print("Blog Post #5: Overfitting and Regularization")
    print("="*70)
    
    # Subtask 7.1: Generate overfitting examples
    generate_overfitting_examples()
    
    # Subtask 7.2: Generate comparison visualizations
    generate_comparison_visualizations()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - visualizations/01_overfitting_no_regularization.png")
    print("  - visualizations/02_no_regularization.png")
    print("  - visualizations/03_dropout_only.png")
    print("  - visualizations/04_weight_decay_only.png")
    print("  - visualizations/05_both_regularization.png")
    print("  - visualizations/06_comparison_all_metrics.png")
    print("\nThese visualizations are ready for use in the blog post.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
