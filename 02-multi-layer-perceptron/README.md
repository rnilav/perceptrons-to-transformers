# Multi-Layer Perceptron - Breaking Through the XOR Barrier

The multi-layer perceptron (MLP) solved the problem that stumped single-layer perceptrons: non-linear classification. By adding hidden layers with non-linear activation functions, MLPs can learn complex decision boundaries and solve problems like XOR that are impossible for single-layer networks.

## 📖 Blog Post

Read the full explanation: [Understanding AI from First Principles: From Lines to Curves - The Hidden Layer](https://dev.to/rnilav/understanding-ai-from-first-principles-multi-layer-perceptrons-and-the-hidden-layer-breakthrough-44pl)

## 🎯 What It Does

MLPs extend perceptrons by adding one or more hidden layers between input and output. This seemingly simple change enables:
- Non-linear classification (solving XOR and beyond)
- Learning complex decision boundaries
- Hierarchical feature learning
- Foundation for modern deep learning

## 📁 Files

- **`mlp.py`** - Core MLP implementation with forward pass
- **`mlp_playground.py`** - Interactive Streamlit app with 4 demos

## 🚀 Quick Start

### Run the Interactive Playground

```bash
streamlit run mlp_playground.py
```

Features:
- **XOR Problem Demo** - See how MLPs solve the classic XOR problem
- **Activation Functions Explorer** - Compare sigmoid, tanh, and ReLU
- **Architecture Builder** - Configure custom network architectures
- **Decision Boundaries** - Visualize non-linear classification

### Use in Your Code

```python
from mlp import MLP, create_xor_network
import numpy as np

# Create a custom MLP
mlp = MLP(
    layer_sizes=[2, 4, 1],  # 2 inputs, 4 hidden, 1 output
    activations=['relu', 'sigmoid'],
    random_state=42
)

# Make predictions
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = mlp.predict(X)

# Or use the pre-configured XOR solver
xor_net = create_xor_network()
xor_output = xor_net.predict(X)
print(xor_output)  # [0.0000, 1.0000, 1.0000, 0.0000]

# View network architecture
print(mlp.summary())
```


## 🧮 How It Works

### Forward Pass

An MLP computes outputs by passing data through multiple layers:

```
Layer 1: z₁ = X @ W₁ + b₁
         a₁ = activation(z₁)

Layer 2: z₂ = a₁ @ W₂ + b₂
         a₂ = activation(z₂)

Output: a₂
```

Each layer applies:
1. Linear transformation (weights and biases)
2. Non-linear activation function

### The XOR Solution

A 2-2-1 network solves XOR by:
- Hidden neuron 1: Learns OR-like pattern (x₁ OR x₂)
- Hidden neuron 2: Learns AND-like pattern (x₁ AND x₂)
- Output: Combines them (OR AND NOT AND = XOR)

## 🔑 Key Concepts

**Hidden Layers** - Layers between input and output that transform the data into representations where problems become linearly separable

**Activation Function** - Non-linear functions (sigmoid) that enable networks to learn complex patterns

**Non-Linear Classification** - Ability to create curved decision boundaries, not just straight lines

**Forward Pass** - Computing outputs by passing inputs through all layers sequentially

**Linear Separability** - Whether classes can be separated by a straight line (or hyperplane)

## ⚠️ The XOR Problem

XOR (exclusive-or) is the classic problem that exposed the limitation of single-layer perceptrons:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

**Why it matters:** 
- Proved single-layer perceptrons have fundamental limitations
- Triggered the first "AI winter" in the 1970s
- Solving it required hidden layers, unlocking modern neural networks
- Demonstrates the power of hierarchical representations

## 🎓 What You'll Learn

1. Why single-layer perceptrons cannot solve XOR
2. How hidden layers enable non-linear classification
3. The role of activation functions in neural networks
4. How to build and use multi-layer networks
5. The geometric intuition behind decision boundaries
6. Foundation concepts for deep learning

## 📚 Next Steps

After mastering MLPs, explore:
- **[03-backpropagation/](../03-backpropagation/)** - How networks learn (coming soon)
- The notebook for deeper mathematical insights
- The playground to build intuition through experimentation
- Try modifying the XOR network weights to understand how it works

## 🔗 Prerequisites

Before diving into MLPs, make sure you understand:
- **[01-perceptron/](../01-perceptron/)** - Single-layer perceptrons and their limitations
- Basic linear algebra (matrix multiplication)
- Basic calculus concepts (for understanding activation functions)

## 🔗 References

1. **Minsky, M., & Papert, S.** (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

2. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*. Available at: http://neuralnetworksanddeeplearning.com/

---

**Note:** This implementation focuses on understanding MLP architecture and forward pass computation. Training (backpropagation) will be covered in the next module.
