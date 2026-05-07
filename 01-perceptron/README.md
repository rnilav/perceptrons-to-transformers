# Perceptron - The First Artificial Neuron

## 📖 Blog Post

Read the full explanation - [Understanding Perceptrons: The Foundation of Modern AI](https://dev.to/rnilav/understanding-perceptrons-the-foundation-of-modern-ai-2g04)

## 🎯 What It Does

The perceptron can learn simple logic gates like AND, OR & NAND by adjusting weights based on training examples. However, it cannot learn XOR - a limitation that held back neural networks for decades.

## 🚀 Quick Start

### Run the Interactive Playground

```bash
streamlit run perceptron_playground.py
```

Features:
- Train on different datasets (AND, OR, XOR, NAND)
- Adjust learning rate and iterations
- Visualize decision boundaries in real-time
- See training progress and convergence
- Understand why XOR fails


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


## 📚 Next Steps

After mastering the perceptron, explore:
- [02-multi-layer-perceptron](https://github.com/rnilav/perceptrons-to-transformers/tree/main/02-multi-layer-perceptron)
- Why XOR fails and how to solve it
- The notebook for deeper mathematical insights
- The playground to build intuition through experimentation