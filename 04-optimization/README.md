# Blog Post #4 - Modern Optimization & Scale

This directory contains the code and interactive playground for the fourth blog post in the "Perceptrons to Transformers" series, focusing on the transition from small-scale learning to large-scale learning with MNIST, introducing stochastic gradient descent, mini-batches, and modern optimizers (Momentum and Adam).

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Navigate to this directory:
```bash
cd perceptrons-to-transformers/04-optimization/
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - For numerical computations
- `matplotlib` - For plotting and visualization
- `streamlit` - For the interactive playground
- `tensorflow` - For MNIST dataset loading only

### Running the Interactive Playground

To launch the Streamlit playground:

```bash
streamlit run optimization_playground.py
```

This will open a browser window with the interactive playground where you can:
- Compare different optimizers (SGD, Momentum, Adam) on MNIST
- Visualize training accuracy curves in real-time
- See performance comparisons and sample predictions

#### Run with Verbose Output

For detailed test output:

```bash
pytest -v
```

### Training a Model Manually

You can train a model directly using the trainer:

```python
from mnist_trainer import load_mnist, Network, Trainer
from optimizers import Adam

# Load data
(X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot) = load_mnist()

# Create network and optimizer
network = Network(hidden_size=128, seed=42)
optimizer = Adam(learning_rate=0.01)

# Train
trainer = Trainer(network, optimizer)
history = trainer.train(
    X_train, y_train_onehot, y_train,
    X_test, y_test_onehot, y_test,
    epochs=5, batch_size=64, verbose=True
)

# Evaluate
test_accuracy = network.accuracy(X_test, y_test)
print(f"Final test accuracy: {test_accuracy:.4f}")
```

## Code Organization

### Core Implementation Files

- **`optimizers.py`** - Optimizer implementations
  - `Optimizer` - Base class with common interface
  - `SGD` - Stochastic gradient descent (mini-batch)
  - `MomentumSGD` - SGD with momentum for smoother convergence
  - `Adam` - Adaptive moment estimation optimizer

- **`mnist_trainer.py`** - Neural network and training infrastructure
  - `load_mnist()` - Load and preprocess MNIST dataset
  - `Network` - Neural network class (784 → 128 → 10)
    - Forward propagation with ReLU and softmax
    - Backward propagation for gradient computation
    - Loss computation and accuracy evaluation
  - `Trainer` - Training loop with mini-batch support
    - Epoch-based training with shuffling
    - Metrics tracking (loss and accuracy)
    - Integration with any optimizer

- **`optimization_playground.py`** - Interactive Streamlit application
  - Optimizer Comparison
    - Side-by-side training with multiple optimizers
    - Real-time loss and accuracy visualization
    - Performance comparison and sample predictions

### Supporting Files

- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation file

## Features

- **From-scratch implementations**: All optimizers built using only NumPy
- **Interactive comparison**: Side-by-side optimizer performance on real MNIST data

## Learning Objectives

1. Understand why full-batch gradient descent doesn't scale to large datasets
2. Grasp the concepts of mini-batches, epochs, and stochastic updates
3. Recognize the problems with vanilla SGD (noise, oscillation, sensitivity)
4. Develop intuition for how Momentum and Adam improve convergence
5. Experience the practical differences through interactive visualization

## Usage Tips

### Playground Recommendations

- **Start with default settings**: Use Adam optimizer with learning rate 0.01, batch size 64, and 5 epochs
- **Compare optimizers**: Select multiple optimizers to see performance differences side-by-side
- **Experiment with hyperparameters**: Try different learning rates and batch sizes to see their effects


## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`
- **Solution**: Run `pip install -r requirements.txt` to install dependencies

**Issue**: Streamlit playground doesn't open in browser
- **Solution**: Check the terminal output for the local URL (usually `http://localhost:8501`) and open it manually

**Issue**: Training is very slow
- **Solution**: Reduce the number of epochs or increase batch size. Ensure NumPy is using optimized BLAS libraries.

**Issue**: `tensorflow` installation fails
- **Solution**: TensorFlow is only used for loading MNIST data. You can use an alternative MNIST loader or download the dataset manually.

## Project Context

This is the fourth post in the "Perceptrons to Transformers" series:
1. **Post 1**: [Perceptron: The Foundation of Modern AI](https://dev.to/rnilav/understanding-perceptrons-the-foundation-of-modern-ai-2g04)
2. **Post 2**: [Multi Layer Perceptron: From Lines to Curves - The Hidden Layer](https://dev.to/rnilav/understanding-ai-from-first-principles-multi-layer-perceptrons-and-the-hidden-layer-breakthrough-44pl)
3. **Post 3**: [Backpropagation: Errors Flow Backward, Knowledge Flows Forward](https://dev.to/rnilav/3-backpropagation-errors-flow-backward-knowledge-flows-forward-5320)
4. **Post 4**: [Neural Network Optimizers: From Baby Steps to Intelligent Learning](TBU)


## License

This code is part of an educational blog series and is provided for learning purposes.
