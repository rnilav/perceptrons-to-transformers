# Blog Post #6: Convolutional Neural Networks (CNNs)

This directory contains the complete implementation of Blog Post #6 from the "From Perceptrons to Transformers" series. This post introduces Convolutional Neural Networks (CNNs), the first major architectural innovation that revolutionized computer vision by leveraging spatial structure and weight sharing.

## Overview

CNNs represent a fundamental shift in how we think about neural networks. Instead of treating all inputs equally (as fully-connected networks do), CNNs exploit the spatial structure of images through:

- **Convolution**: Sliding learnable filters across the image to detect local patterns
- **Pooling**: Reducing spatial dimensions while preserving important information
- **Weight Sharing**: Using the same filter weights across all spatial locations
- **Hierarchical Representation**: Building complex features from simple ones

This module implements all these concepts from scratch using NumPy, providing clear, educational implementations that prioritize understanding over performance.

## Directory Structure

```
06-cnn/
├── convolution.py              # 2D convolution implementation
├── pooling.py                  # Max and average pooling
├── cnn.py                      # Complete CNN architecture
├── cnn_playground.py           # Interactive Streamlit app (2 tabs)
├── README.md                   # This file
├── requirements.txt            # Dependencies
```

### Interactive and Demo Files

**cnn_playground.py**
Interactive Streamlit application with 2 tabs:

1. **FC Network vs CNN** — Train both models on the same 1,000-sample MNIST subset using pure NumPy + Adam (same stack as Post 4). Adjust FC hidden size, CNN filter count, epochs (up to 20), and batch size. After training: live loss/accuracy curves side by side, epoch-by-epoch breakdown table, and parameter count comparison. With only 1,000 samples the CNN consistently outperforms the FC network — weight sharing gives it a structural advantage when data is scarce.

2. **CNN Layer Explorer** — Pick a digit (0, 1, 6, or 8) and explore three views:
   - *What each filter detects*: filter weights (3×3 grid) alongside the response heatmap on the digit
   - *Layer-by-layer pipeline*: traces the digit through Conv1+ReLU → MaxPool → Conv2+ReLU → MaxPool → Flatten → FC → Softmax, with actual feature map images and a dimension table at each stage
   - *MaxPool zoom-in*: shows a 4×4 patch of conv output with actual values, then the 2×2 result after pooling

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```


### 2. Run Interactive Playground

```bash
streamlit run cnn_playground.py
```

Then open your browser to `http://localhost:8501`


## Key Concepts

### Convolution Operation

The convolution operation slides a learnable filter (kernel) across the input image, computing element-wise products and summing the results at each position. This creates a feature map that detects local patterns.

**Output dimensions formula:**
```
H_out = (H_in - K + 2*P) / S + 1
W_out = (W_in - K + 2*P) / S + 1
```

Where:
- H_in, W_in: Input height and width
- K: Kernel size
- P: Padding
- S: Stride

### Pooling Operations

Pooling reduces spatial dimensions while preserving important information:
- **Max Pooling**: Takes the maximum value in each window
- **Average Pooling**: Takes the average value in each window

**Output dimensions formula:**
```
H_out = (H_in - P) / S + 1
W_out = (W_in - P) / S + 1
```

Where:
- H_in, W_in: Input height and width
- P: Pool size
- S: Stride

### CNN Architecture

A typical CNN combines:
1. **Convolution layers**: Learn filters to detect features
2. **Pooling layers**: Reduce spatial dimensions
3. **Fully-connected layers**: Classify based on learned features

### Inductive Bias

CNNs introduce the concept of **inductive bias**—built-in assumptions about the problem structure:
- **Spatial locality**: Nearby pixels are more related than distant pixels
- **Weight sharing**: Same filter weights across all spatial locations
- **Translation invariance**: Features can be detected anywhere in the image

## Quick Start Example

```python
import numpy as np
from cnn import create_simple_cnn

# Create a simple CNN for MNIST-like tasks
cnn = create_simple_cnn(input_shape=(28, 28, 1), num_classes=10)

# Create sample input batch
X = np.random.randn(4, 28, 28, 1)

# Forward pass
logits = cnn.forward(X)
print(f"Output shape: {logits.shape}")  # (4, 10)

# Make predictions
predictions = cnn.predict(X)
print(f"Predictions: {predictions}")  # [3, 7, 2, 9]

# Count parameters
params = cnn.count_parameters()
print(f"Total parameters: {params}")  # ~100K
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the correct directory:
```bash
cd 06-cnn
python -c "import convolution; print('OK')"
```

### Streamlit Issues

If Streamlit doesn't start, try:
```bash
streamlit run cnn_playground.py --logger.level=debug
```

### Test Failures

If tests fail, check that all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## License

This code is part of the "From Perceptrons to Transformers" educational series.
