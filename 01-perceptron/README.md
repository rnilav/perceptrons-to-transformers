# Perceptron - The First Artificial Neuron

The perceptron, invented by Frank Rosenblatt in 1958, was the first artificial neuron that could learn from examples. It's a binary classifier that learns to separate data using a linear decision boundary.

## 📖 Blog Post

Read the full explanation - [Understanding Perceptrons: The Foundation of Modern AI](https://dev.to/rnilav/understanding-perceptrons-the-foundation-of-modern-ai-2g04)

## 🎯 What It Does

The perceptron can learn simple logic gates like AND, OR & NAND by adjusting weights based on training examples. However, it cannot learn XOR - a limitation that held back neural networks for decades.

## 📁 Files

- **`perceptron.py`** - Core implementation with training algorithm
- **`perceptron_playground.py`** - Interactive Streamlit app

## 🚀 Quick Start

### Run the Interactive Playground

```bash
streamlit run perceptron_playground.py
```

Features:
- Train on different datasets (AND, OR, XOR, NAND, random data)
- Adjust learning rate and iterations
- Visualize decision boundaries in real-time
- See training progress and convergence
- Understand why XOR fails

### Use in Your Code

```python
from perceptron import Perceptron
import numpy as np

# Create a perceptron with 2 inputs
p = Perceptron(learning_rate=0.1, n_iterations=100)

# Train on AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
p.fit(X, y)

# Make predictions
predictions = p.predict(X)
print(predictions)  # [0, 0, 0, 1]

# Check accuracy
accuracy = p.score(X, y)
print(f"Accuracy: {accuracy:.1%}")  # 100%
```

## 🧮 How It Works

### Forward Pass

The perceptron computes a weighted sum of inputs plus a bias:

```
z = w₁x₁ + w₂x₂ + b
ŷ = 1 if z ≥ 0, else 0
```

### Learning Rule

When the prediction is wrong, adjust weights:

```
w ← w + α(y - ŷ)x
b ← b + α(y - ŷ)
```

Where α is the learning rate.

## 🔑 Key Concepts

**Weights** - Determine importance of each input (learned from data)

**Bias** - Shifts the decision boundary away from origin

**Learning Rate** - Step size for weight updates (higher = faster but less stable)

**Linear Separability** - Data must be separable by a straight line

**Convergence** - Perceptron converges if data is linearly separable

## ⚠️ Limitations

The perceptron can only learn **linearly separable** functions:
- ✅ AND, OR, NAND - converges perfectly
- ❌ XOR - cannot learn (not linearly separable)

This limitation led to the development of multilayer perceptrons (MLPs) in the next module.

## 🎓 What You'll Learn

1. How perceptron learn from examples
2. The role of weights and biases
3. Why linear separability matters
4. The XOR problem and its historical significance
5. Foundation for understanding modern neural networks

## 📚 Next Steps

After mastering the perceptron, explore:
- [**02-multi-layer-perceptron**](https://github.com/rnilav/perceptrons-to-transformers/tree/main/02-multi-layer-perceptron)
- Why XOR fails and how to solve it
- The notebook for deeper mathematical insights
- The playground to build intuition through experimentation