"""Interactive Hyperparameter Exploration Script.

This script demonstrates how different hyperparameters affect training.
Run this to build intuition about neural network training.

See HYPERPARAMETER_INSIGHTS.md for detailed explanations.
"""

import numpy as np
from backprop import TrainableMLP
import time

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


def train_and_report(architecture, learning_rate, random_state, epochs, label):
    """Train a network and report results."""
    print(f"\n{'='*70}")
    print(f"Configuration: {label}")
    print(f"{'='*70}")
    print(f"Architecture:   {architecture}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Random Seed:    {random_state}")
    print(f"Epochs:         {epochs}")
    print("-" * 70)
    
    # Create and train
    mlp = TrainableMLP(
        layer_sizes=architecture,
        activations=['sigmoid', 'sigmoid'],
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    start_time = time.time()
    history = mlp.train(X, y, epochs=epochs, verbose=False)
    train_time = time.time() - start_time
    
    # Evaluate
    predictions = mlp.predict(X)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)
    
    # Report
    print(f"Training Time:  {train_time:.2f}s")
    print(f"Initial Loss:   {history['loss'][0]:.6f}")
    print(f"Final Loss:     {history['loss'][-1]:.6f}")
    print(f"Loss Reduction: {(1 - history['loss'][-1]/history['loss'][0]) * 100:.1f}%")
    print(f"Final Accuracy: {accuracy * 100:.0f}%")
    
    # Show predictions
    print("\nPredictions:")
    for i, (x_val, y_true, y_pred) in enumerate(zip(X, y, predictions)):
        status = "✓" if (y_pred[0] > 0.5) == y_true[0] else "✗"
        print(f"  {status} [{x_val[0]}, {x_val[1]}] → True: {y_true[0]}, Pred: {y_pred[0]:.4f}")
    
    # Convergence assessment
    if accuracy == 1.0:
        print("\n✅ SUCCESS: Network converged to 100% accuracy")
    elif accuracy >= 0.75:
        print("\n⚠️  PARTIAL: Network learned some patterns but not all")
    else:
        print("\n❌ FAILURE: Network did not learn effectively")
    
    return history, accuracy


def main():
    """Run hyperparameter exploration experiments."""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER EXPLORATION FOR XOR PROBLEM")
    print("="*70)
    print("\nThis script demonstrates how different hyperparameters affect training.")
    print("See HYPERPARAMETER_INSIGHTS.md for detailed explanations.\n")
    
    # ========================================================================
    # EXPERIMENT 1: Learning Rate Effects
    # ========================================================================
    print("\n" + "#"*70)
    print("# EXPERIMENT 1: Learning Rate Effects")
    print("#"*70)
    print("\nQuestion: How does learning rate affect convergence speed?")
    print("Setup: 2-2-1 architecture, seed=123, 5000 epochs")
    
    input("\nPress Enter to start Experiment 1...")
    
    for lr in [0.1, 0.3, 0.5]:
        train_and_report(
            architecture=[2, 2, 1],
            learning_rate=lr,
            random_state=123,
            epochs=5000,
            label=f"Learning Rate = {lr}"
        )
        time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("INSIGHT: Higher learning rates converge faster but may be less stable.")
    print("         LR=0.3-0.5 provides good balance for this problem.")
    
    # ========================================================================
    # EXPERIMENT 2: Architecture Capacity
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 2: Architecture Capacity")
    print("#"*70)
    print("\nQuestion: How does network size affect robustness?")
    print("Setup: Learning rate=0.5, seed=42 (known difficult seed), 5000 epochs")
    
    input("\nPress Enter to start Experiment 2...")
    
    for arch in [[2, 2, 1], [2, 4, 1], [2, 8, 1]]:
        arch_str = f"{arch[0]}-{arch[1]}-{arch[2]}"
        train_and_report(
            architecture=arch,
            learning_rate=0.5,
            random_state=42,  # Difficult seed for 2-2-1
            epochs=5000,
            label=f"Architecture = {arch_str}"
        )
        time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("INSIGHT: Larger networks are more robust to bad initializations.")
    print("         2-2-1 may struggle with seed=42, but 2-4-1 handles it well.")
    
    # ========================================================================
    # EXPERIMENT 3: Random Seed Sensitivity
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 3: Random Seed Sensitivity")
    print("#"*70)
    print("\nQuestion: How does weight initialization affect results?")
    print("Setup: 2-2-1 architecture, learning rate=0.5, 5000 epochs")
    
    input("\nPress Enter to start Experiment 3...")
    
    results = []
    for seed in [42, 123, 456, 789]:
        history, accuracy = train_and_report(
            architecture=[2, 2, 1],
            learning_rate=0.5,
            random_state=seed,
            epochs=5000,
            label=f"Random Seed = {seed}"
        )
        results.append((seed, accuracy, history['loss'][-1]))
        time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("SEED COMPARISON:")
    print("-" * 70)
    for seed, acc, loss in results:
        status = "✅" if acc == 1.0 else "⚠️" if acc >= 0.75 else "❌"
        print(f"  {status} Seed {seed:3d}: Accuracy={acc*100:3.0f}%, Final Loss={loss:.6f}")
    
    print("\n" + "-"*70)
    print("INSIGHT: Some random seeds lead to better starting points than others.")
    print("         This is why larger networks or multiple training runs help.")
    
    # ========================================================================
    # EXPERIMENT 4: Epoch Requirements
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# EXPERIMENT 4: Epoch Requirements")
    print("#"*70)
    print("\nQuestion: How many epochs are actually needed?")
    print("Setup: 2-2-1 architecture, learning rate=0.3, seed=123")
    
    input("\nPress Enter to start Experiment 4...")
    
    for epochs in [1000, 3000, 5000, 10000]:
        train_and_report(
            architecture=[2, 2, 1],
            learning_rate=0.3,
            random_state=123,
            epochs=epochs,
            label=f"Epochs = {epochs}"
        )
        time.sleep(0.5)
    
    print("\n" + "-"*70)
    print("INSIGHT: More epochs help up to a point, then provide diminishing returns.")
    print("         Watch the loss curve to know when to stop training.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXPLORATION COMPLETE!")
    print("="*70)
    print("\nKEY TAKEAWAYS:")
    print("  1. Learning rate (0.3-0.5) is often the most important hyperparameter")
    print("  2. Larger networks (2-4-1) are more robust than minimal ones (2-2-1)")
    print("  3. Random initialization matters - some seeds work better than others")
    print("  4. Training time depends on learning rate and architecture")
    print("  5. Watch the loss curve to know when training has converged")
    print("\nRECOMMENDED CONFIGURATIONS:")
    print("  • For learning:  2-2-1, LR=0.3, seed=123, 5000 epochs")
    print("  • For robustness: 2-4-1, LR=0.5, seed=123, 3000 epochs")
    print("\nFor more details, see HYPERPARAMETER_INSIGHTS.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
