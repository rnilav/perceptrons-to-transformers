# Backpropagation: How Neural Networks Learn

This directory contains the implementation and educational materials for Blog Post 3 in the [From perceptrons to transformers](https://dev.to/rnilav/understanding-perceptrons-the-foundation-of-modern-ai-2g04) series

## Overview

Backpropagation is the algorithm that enables neural networks to learn automatically by computing gradients and updating weights through gradient descent. This implementation demonstrates the complete training process on the XOR problem.

## Files

### Core Implementation
- **`backprop.py`** - Main implementation with `TrainableMLP` class
  - Extends the MLP class from Post 2 with training capabilities
  - Implements backpropagation algorithm for gradient computation
  - Implements gradient descent for weight updates
  - Includes activation derivative functions

### Documentation
- **`HYPERPARAMETER_INSIGHTS.md`** - Comprehensive guide to hyperparameter effects
  - Learning rate selection and effects
  - Architecture capacity and robustness
  - Random seed sensitivity
  - Epoch requirements
  - Troubleshooting guide
  - Recommended configurations

### Examples and Exploration
- **`example_xor_training.py`** - Simple example showing XOR training
  - Quick demonstration of the training process
  - Shows loss reduction and final predictions
  - Good starting point for beginners

- **`explore_hyperparameters.py`** - Interactive exploration script
  - 4 experiments demonstrating hyperparameter effects
  - Compares different configurations side-by-side
  - Provides insights and recommendations
  - Run this to build intuition!

### Interactive Playground (Coming Soon)
- **`backprop_playground.py`** - Streamlit app with 2 tabs:
  - Tab 1: Training Visualization (watch XOR learning in real-time)
  - Tab 2: Gradient Flow Visualization (see backpropagation in action)


## Quick Start

### 1. Basic Training Example
```bash
python example_xor_training.py
```

This shows a simple training run with recommended settings.

### 2. Explore Hyperparameters
```bash
python explore_hyperparameters.py
```

This runs 4 experiments showing how different hyperparameters affect training. Great for building intuition!

### 3. Use in Your Code
```python
from backprop import TrainableMLP
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
mlp = TrainableMLP(
    layer_sizes=[2, 4, 1],
    activations=['sigmoid', 'sigmoid'],
    learning_rate=0.5,
    random_state=123
)

history = mlp.train(X, y, epochs=3000, verbose=True)

# Make predictions
predictions = mlp.predict(X)
print(predictions)
```

## Recommended Configurations

### For Learning and Understanding
```python
TrainableMLP(
    layer_sizes=[2, 2, 1],      # Minimal architecture
    activations=['sigmoid', 'sigmoid'],
    learning_rate=0.3,           # Balanced learning rate
    random_state=123             # Known good seed
)
epochs = 5000
```
**Why:** Demonstrates minimal requirements, educational value

### For Reliable Results
```python
TrainableMLP(
    layer_sizes=[2, 4, 1],      # More robust
    activations=['sigmoid', 'sigmoid'],
    learning_rate=0.5,           # Faster convergence
    random_state=123             # Known good seed
)
epochs = 3000
```
**Why:** Robust, fast convergence, works with most seeds

## Key Concepts Demonstrated

1. **Backpropagation Algorithm**
   - Forward pass: Compute predictions
   - Loss computation: Measure error (MSE)
   - Backward pass: Compute gradients using chain rule
   - Weight update: Apply gradient descent

2. **Gradient Descent**
   - Formula: `w_new = w_old - learning_rate * gradient`
   - Iteratively minimizes loss function
   - Learning rate controls step size

3. **Hyperparameter Effects**
   - Learning rate: Speed vs. stability tradeoff
   - Architecture: Capacity vs. complexity tradeoff
   - Random seed: Initialization matters
   - Epochs: When to stop training

## Understanding the Loss Curve

### Good Training (Converging)
```
Epoch    0: Loss = 0.250
Epoch 1000: Loss = 0.100
Epoch 2000: Loss = 0.020
Epoch 3000: Loss = 0.005
```
✅ Loss decreases smoothly → Network is learning

### Stuck in Local Minimum
```
Epoch    0: Loss = 0.250
Epoch 1000: Loss = 0.240
Epoch 5000: Loss = 0.235
```
⚠️ Loss barely changes → Try different seed or increase learning rate

### Unstable Training
```
Epoch    0: Loss = 0.250
Epoch  100: Loss = 0.150
Epoch  200: Loss = 0.350
Epoch  300: Loss = 0.100
```
❌ Loss oscillates → Learning rate too high

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Accuracy stuck at 50% | Try different random seed or increase learning rate |
| Accuracy stuck at 75% | Increase network capacity (2-4-1) or try different seed |
| Loss increases | Decrease learning rate |
| Loss decreases very slowly | Increase learning rate |
| Different runs give different results | Use larger network or average multiple runs |

## Next Steps

After understanding backpropagation:
1. Explore the interactive playground (coming soon)
2. Read `HYPERPARAMETER_INSIGHTS.md` for deep dive
3. Run `explore_hyperparameters.py` to build intuition
4. Experiment with different configurations
5. Move on to optimization algorithms (SGD, momentum, Adam)

## Dependencies

- NumPy (numerical computations)
- Streamlit (for playground, optional)
- Matplotlib (for visualizations, optional)

## Educational Notes

This implementation prioritizes clarity and understanding over performance:
- Uses NumPy only (no ML frameworks)
- Includes detailed comments and docstrings
- Demonstrates concepts with minimal code
- Focuses on the XOR problem for simplicity
