# Blog Post #5: Overfitting and Regularization

This directory contains the implementation for the fifth blog post in the "Perceptrons to Transformers" series, focusing on overfitting and regularization techniques.

## Overview

This blog post demonstrates why neural networks overfit (memorize training data instead of learning generalizable patterns) and introduces two key regularization techniques:

- **Dropout**: Randomly deactivates neurons during training to prevent co-adaptation
- **Weight Decay**: Penalizes large weights to encourage simpler models

The interactive playground lets you experiment with these techniques in real-time, visualizing how they reduce the gap between training and test accuracy. Please note the training time taken is proportional to the number of epochs and the network size (it would range between 30 secs - 5 mins)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

Navigate to this directory and install required packages:

```bash
cd blogpost-perceptrons-to-transformers/05-regularization
pip install -r requirements.txt
```

This installs:
- **numpy**: Numerical computations
- **matplotlib**: Visualization and plotting
- **streamlit**: Interactive web app framework
- **tensorflow**: MNIST dataset loading

### Step 2: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import numpy, matplotlib, streamlit, tensorflow; print('All dependencies installed successfully!')"
```

## Quick Start

### Option 1: Interactive Playground (Recommended)

Launch the interactive Streamlit app to experiment with regularization parameters:

```bash
streamlit run regularization_playground.py
```

This opens a web browser with two tabs:

**Tab 1: Overfitting Demonstration**
- Adjust dropout rate (0.0 to 0.9)
- Adjust weight decay coefficient (0.0 to 0.01)
- Choose network size (128, 256, or 512 hidden units)
- Set number of epochs and batch size
- Click "Train Network" to see real-time results
- Visualizations show:
  - Train vs test accuracy curves
  - Loss decomposition (cross-entropy + penalty)
  - Overfitting gap over time
  - Final performance metrics

**Tab 2: Weight Distribution Analysis**
- Compare weight distributions with different regularization settings
- See how weight decay keeps weights smaller
- Understand the effect of dropout on weight magnitudes

### Option 2: Overfitting Demonstration Script

Run a pre-configured demonstration showing overfitting effects:

```bash
python overfitting_demo.py
```

This generates visualizations showing:
- Networks without regularization (high overfitting)
- Networks with dropout only
- Networks with weight decay only
- Networks with both techniques combined

Results are saved to the `visualizations/` directory.

### Key Components

#### `regularization.py`
Implements core regularization techniques:
- **Dropout class**: Forward and backward passes with random neuron deactivation
- **Weight decay functions**: Penalty computation and gradient application

#### `network_with_regularization.py`
Extends the network from blog post #4:
- **RegularizedNetwork**: Network class with dropout and weight decay support
- **RegularizedTrainer**: Training loop that handles regularization
- Methods for training/inference mode switching
- Loss decomposition (CE loss + penalty)

#### `regularization_playground.py`
Interactive Streamlit application:
- Real-time parameter adjustment
- Live training visualization
- Multiple comparison modes
- Performance metrics display

#### `overfitting_demo.py`
Standalone demonstration script:
- Pre-configured training scenarios
- Automatic visualization generation
- Comparison of regularization techniques


## Experimental Suggestions

Try these experiments in the playground:

1. **Baseline Overfitting**: Set dropout=0, weight_decay=0, train for 30 epochs. Observe the overfitting gap grow.

2. **Dropout Effect**: Set dropout=0.5, weight_decay=0. See how dropout reduces the gap.

3. **Weight Decay Effect**: Set dropout=0, weight_decay=0.001. See how weight decay helps.

4. **Combined Effect**: Set dropout=0.5, weight_decay=0.001. See the best results.

5. **Network Size**: Try different hidden sizes (128, 256, 512) with same regularization. Larger networks overfit more.

6. **Epoch Sensitivity**: Train for 5, 10, 20, 50 epochs. See how overfitting increases with training time.

## Troubleshooting

### Streamlit App Won't Start

```bash
# Clear Streamlit cache
streamlit cache clear

# Try running with explicit port
streamlit run regularization_playground.py --server.port 8501
```

### Out of Memory Error

Reduce batch size or network size:
```bash
# In the playground, try:
- Batch Size: 32 (instead of 64 or 128)
- Network Size: 128 (instead of 256 or 512)
```

### Tests Fail

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Performance Notes

- Training 10 epochs on MNIST typically takes 2-5 seconds
- Playground is responsive for real-time parameter adjustment
- Weight distribution analysis takes ~30 seconds for 4 configurations
- All computations use NumPy