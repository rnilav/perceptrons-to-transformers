"""
Pooling operations for CNNs.

This module implements pooling operations from scratch using NumPy, including:
- max_pool_2d(): Maximum pooling operation
- average_pool_2d(): Average pooling operation
- visualize_pooling_step(): Visualization of the pooling process

All implementations use NumPy only for educational clarity.
"""

from typing import Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Type aliases
NDArray = np.ndarray


def max_pool_2d(
    input_data: NDArray,
    pool_size: int = 2,
    stride: int = 2
) -> NDArray:
    """
    Apply max pooling to input data.
    
    Max pooling reduces spatial dimensions by taking the maximum value
    in each pooling window. This operation:
    - Reduces spatial dimensions
    - Reduces parameters (no learnable weights)
    - Provides translation invariance
    
    Args:
        input_data: Input feature map of shape (H, W) or (H, W, C)
        pool_size: Size of the pooling window (default: 2)
        stride: Number of pixels to move the window (default: 2)
    
    Returns:
        Pooled feature map of shape (H_out, W_out) where:
        H_out = (H - pool_size) / stride + 1
        W_out = (W - pool_size) / stride + 1
    
    Raises:
        ValueError: If pool_size <= 0 or stride <= 0
    
    Example:
        >>> feature_map = np.arange(16).reshape(4, 4)
        >>> output = max_pool_2d(feature_map, pool_size=2, stride=2)
        >>> output.shape
        (2, 2)
    """
    # Validate inputs
    if pool_size <= 0:
        raise ValueError(f"Pool size must be positive integer, got {pool_size}")
    
    if stride <= 0:
        raise ValueError(f"Stride must be positive integer, got {stride}")
    
    # Handle 2D vs 3D inputs
    if input_data.ndim == 2:
        input_data = input_data[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    H, W, C = input_data.shape
    
    # Calculate output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    # Initialize output feature map
    output = np.zeros((H_out, W_out, C))
    
    # Perform max pooling: slide window across input
    for i in range(H_out):
        for j in range(W_out):
            # Extract window from input
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            window = input_data[h_start:h_end, w_start:w_end, :]
            
            # Take maximum value in window for each channel
            output[i, j, :] = np.max(window.reshape(-1, C), axis=0)
    
    # Remove channel dimension if input was 2D
    if squeeze_output:
        output = output[:, :, 0]
    
    return output


def average_pool_2d(
    input_data: NDArray,
    pool_size: int = 2,
    stride: int = 2
) -> NDArray:
    """
    Apply average pooling to input data.
    
    Average pooling reduces spatial dimensions by taking the mean value
    in each pooling window. This operation:
    - Reduces spatial dimensions
    - Reduces parameters (no learnable weights)
    - Smooths feature maps
    
    Args:
        input_data: Input feature map of shape (H, W) or (H, W, C)
        pool_size: Size of the pooling window (default: 2)
        stride: Number of pixels to move the window (default: 2)
    
    Returns:
        Pooled feature map of shape (H_out, W_out) where:
        H_out = (H - pool_size) / stride + 1
        W_out = (W - pool_size) / stride + 1
    
    Raises:
        ValueError: If pool_size <= 0 or stride <= 0
    
    Example:
        >>> feature_map = np.arange(16).reshape(4, 4).astype(float)
        >>> output = average_pool_2d(feature_map, pool_size=2, stride=2)
        >>> output.shape
        (2, 2)
    """
    # Validate inputs
    if pool_size <= 0:
        raise ValueError(f"Pool size must be positive integer, got {pool_size}")
    
    if stride <= 0:
        raise ValueError(f"Stride must be positive integer, got {stride}")
    
    # Handle 2D vs 3D inputs
    if input_data.ndim == 2:
        input_data = input_data[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    H, W, C = input_data.shape
    
    # Calculate output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    # Initialize output feature map
    output = np.zeros((H_out, W_out, C))
    
    # Perform average pooling: slide window across input
    for i in range(H_out):
        for j in range(W_out):
            # Extract window from input
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            window = input_data[h_start:h_end, w_start:w_end, :]
            
            # Take mean value in window for each channel
            output[i, j, :] = np.mean(window.reshape(-1, C), axis=0)
    
    # Remove channel dimension if input was 2D
    if squeeze_output:
        output = output[:, :, 0]
    
    return output


def visualize_pooling_step(
    input_data: NDArray,
    pool_size: int = 2,
    stride: int = 2,
    pool_type: Literal["max", "average"] = "max",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize the pooling process step-by-step.
    
    Creates a visualization showing:
    - Input feature map with grid overlay
    - Pooling window sliding across the input
    - Value selection (max or average) at each position
    - Resulting pooled feature map
    
    Args:
        input_data: Input feature map of shape (H, W) or (H, W, C)
        pool_size: Size of the pooling window (default: 2)
        stride: Stride parameter (default: 2)
        pool_type: Type of pooling - "max" or "average" (default: "max")
        figsize: Figure size for matplotlib (default: (12, 8))
    
    Returns:
        None (displays plot)
    
    Raises:
        ValueError: If pool_type is not "max" or "average"
    
    Example:
        >>> feature_map = np.arange(16).reshape(4, 4).astype(float)
        >>> visualize_pooling_step(feature_map, pool_size=2, stride=2)
    """
    # Validate pool_type
    if pool_type not in ["max", "average"]:
        raise ValueError(f"pool_type must be 'max' or 'average', got {pool_type}")
    
    # Handle 2D vs 3D inputs
    if input_data.ndim == 3:
        # For multi-channel, use first channel for visualization
        input_2d = input_data[:, :, 0]
    else:
        input_2d = input_data
    
    # Apply pooling
    if pool_type == "max":
        output = max_pool_2d(input_data, pool_size=pool_size, stride=stride)
    else:
        output = average_pool_2d(input_data, pool_size=pool_size, stride=stride)
    
    # Handle output shape if it's 3D
    if output.ndim == 3:
        output_2d = output[:, :, 0]
    else:
        output_2d = output
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{pool_type.capitalize()} Pooling Visualization', fontsize=14, fontweight='bold')
    
    # Plot 1: Input feature map
    ax = axes[0, 0]
    im1 = ax.imshow(input_2d, cmap='viridis')
    ax.set_title('Input Feature Map')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im1, ax=ax)
    
    # Add grid to show pixel boundaries
    for i in range(input_2d.shape[0] + 1):
        ax.axhline(i - 0.5, color='red', linewidth=0.5, alpha=0.3)
    for j in range(input_2d.shape[1] + 1):
        ax.axvline(j - 0.5, color='red', linewidth=0.5, alpha=0.3)
    
    # Plot 2: Pooling window at first position
    ax = axes[0, 1]
    window = input_2d[:pool_size, :pool_size]
    im2 = ax.imshow(window, cmap='viridis')
    ax.set_title(f'Pooling Window at Position (0,0)\nSize: {pool_size}×{pool_size}')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im2, ax=ax)
    
    # Add text annotations showing values
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            ax.text(j, i, f'{window[i, j]:.1f}', 
                   ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    # Plot 3: Example pooling operation
    ax = axes[1, 0]
    window_flat = window.flatten()
    
    if pool_type == "max":
        selected_value = np.max(window_flat)
        operation_text = f'Max = {selected_value:.1f}'
    else:
        selected_value = np.mean(window_flat)
        operation_text = f'Average = {selected_value:.1f}'
    
    # Create bar chart showing values in window
    bars = ax.bar(range(len(window_flat)), window_flat, color='steelblue', alpha=0.7)
    
    # Highlight the selected value
    if pool_type == "max":
        max_idx = np.argmax(window_flat)
        bars[max_idx].set_color('red')
    
    ax.set_title(f'{pool_type.capitalize()} Pooling Operation\n{operation_text}')
    ax.set_xlabel('Window Element Index')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, window_flat)):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
               ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Output feature map
    ax = axes[1, 1]
    im4 = ax.imshow(output_2d, cmap='viridis')
    ax.set_title('Output Feature Map')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im4, ax=ax)
    
    # Add text showing dimensions
    fig.text(0.5, 0.02, 
            f'Input: {input_2d.shape} | Pool Size: {pool_size}×{pool_size} | Stride: {stride} | Output: {output_2d.shape}',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
