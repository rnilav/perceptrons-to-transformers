"""
Convolution operations for CNNs.

This module implements 2D convolution from scratch using NumPy, including:
- convolve_2d(): Core 2D convolution operation with stride and padding
- create_demo_filters(): Hand-crafted filters for demonstration
- visualize_convolution_step(): Visualization of the convolution process

All implementations use NumPy only for educational clarity.
"""

from typing import Dict, Tuple, Optional
import numpy as np

# Type aliases
NDArray = np.ndarray


def convolve_2d(
    input_data: NDArray,
    kernel: NDArray,
    stride: int = 1,
    padding: int = 0
) -> NDArray:
    """
    Apply 2D convolution to input data.
    
    This function implements the core convolution operation: sliding a kernel
    (filter) across the input, computing element-wise products, and summing
    the results at each position.
    
    Args:
        input_data: Input image of shape (H, W) or (H, W, C)
        kernel: Convolution kernel of shape (K, K) or (K, K, C)
        stride: Number of pixels to move the kernel at each step (default: 1)
        padding: Number of zeros to add around the input border (default: 0)
    
    Returns:
        Feature map of shape (H_out, W_out) where:
        H_out = (H - K + 2*padding) / stride + 1
        W_out = (W - K + 2*padding) / stride + 1
    
    Raises:
        ValueError: If stride <= 0, kernel size is even, or padding < 0
        ValueError: If input and kernel have incompatible shapes
    
    Example:
        >>> input_img = np.random.randn(5, 5)
        >>> kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        >>> output = convolve_2d(input_img, kernel, stride=1, padding=0)
        >>> output.shape
        (3, 3)
    """
    # Validate inputs
    if stride <= 0:
        raise ValueError(f"Stride must be positive integer, got {stride}")
    
    if padding < 0:
        raise ValueError(f"Padding must be non-negative integer, got {padding}")
    
    # Get kernel size (must be square)
    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    if kernel_h != kernel_w:
        raise ValueError(f"Kernel must be square, got shape {kernel.shape}")
    
    if kernel_h % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel_h}")
    
    # Handle 2D vs 3D inputs
    if input_data.ndim == 2:
        input_data = input_data[:, :, np.newaxis]
    if kernel.ndim == 2:
        kernel = kernel[:, :, np.newaxis]
    
    # Validate channel compatibility
    if input_data.shape[2] != kernel.shape[2]:
        raise ValueError(
            f"Input channels {input_data.shape[2]} must match kernel channels {kernel.shape[2]}"
        )
    
    H, W, C = input_data.shape
    K = kernel_h
    
    # Add padding to input
    if padding > 0:
        padded_input = np.pad(
            input_data,
            ((padding, padding), (padding, padding), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_input = input_data
    
    # Calculate output dimensions
    H_padded = padded_input.shape[0]
    W_padded = padded_input.shape[1]
    
    H_out = (H_padded - K) // stride + 1
    W_out = (W_padded - K) // stride + 1
    
    # Initialize output feature map
    output = np.zeros((H_out, W_out))
    
    # Perform convolution: slide kernel across input
    for i in range(H_out):
        for j in range(W_out):
            # Extract window from padded input
            h_start = i * stride
            h_end = h_start + K
            w_start = j * stride
            w_end = w_start + K
            
            window = padded_input[h_start:h_end, w_start:w_end, :]
            
            # Element-wise multiply and sum
            output[i, j] = np.sum(window * kernel)
    
    return output


def create_demo_filters() -> Dict[str, NDArray]:
    """
    Create hand-crafted filters for demonstration purposes.
    
    Returns a dictionary of filters commonly used in image processing:
    - 'edge_horizontal': Detects horizontal edges (Sobel-like)
    - 'edge_vertical': Detects vertical edges (Sobel-like)
    - 'blur': Smoothing/averaging filter
    - 'sharpen': Sharpening filter
    
    All filter values are in the range [-1, 1] for educational clarity.
    
    Returns:
        Dictionary mapping filter names to 3x3 filter arrays
    
    Example:
        >>> filters = create_demo_filters()
        >>> filters['edge_horizontal'].shape
        (3, 3)
        >>> np.all(filters['edge_horizontal'] >= -1)
        True
        >>> np.all(filters['edge_horizontal'] <= 1)
        True
    """
    filters = {
        # Horizontal edge detection (Sobel-like)
        # Detects transitions in the vertical direction
        'edge_horizontal': np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32) / 8.0,
        
        # Vertical edge detection (Sobel-like)
        # Detects transitions in the horizontal direction
        'edge_vertical': np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ], dtype=np.float32) / 8.0,
        
        # Blur filter (averaging)
        # Smooths the image by averaging neighboring pixels
        'blur': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32) / 9.0,
        
        # Sharpening filter
        # Enhances edges by subtracting blurred version from original
        'sharpen': np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32) / 5.0,
    }
    
    return filters


def visualize_convolution_step(
    input_data: NDArray,
    kernel: NDArray,
    stride: int = 1,
    padding: int = 0,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize the convolution process step-by-step.
    
    Creates a visualization showing:
    - Input image with grid overlay
    - Filter sliding across the input
    - Element-wise multiplication at a specific position
    - Resulting feature map
    
    Args:
        input_data: Input image of shape (H, W) or (H, W, C)
        kernel: Convolution kernel of shape (K, K) or (K, K, C)
        stride: Stride parameter (default: 1)
        padding: Padding parameter (default: 0)
        figsize: Figure size for matplotlib (default: (12, 8))
    
    Returns:
        None (displays plot)
    
    Example:
        >>> input_img = np.random.randn(5, 5)
        >>> kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        >>> visualize_convolution_step(input_img, kernel)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Handle 2D vs 3D inputs
    if input_data.ndim == 3:
        # For multi-channel, use first channel for visualization
        input_2d = input_data[:, :, 0]
    else:
        input_2d = input_data
    
    if kernel.ndim == 3:
        kernel_2d = kernel[:, :, 0]
    else:
        kernel_2d = kernel
    
    # Compute output
    output = convolve_2d(input_data, kernel, stride=stride, padding=padding)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Convolution Operation Visualization', fontsize=14, fontweight='bold')
    
    # Plot 1: Input image
    ax = axes[0, 0]
    im1 = ax.imshow(input_2d, cmap='gray')
    ax.set_title('Input Image')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im1, ax=ax)
    
    # Add grid to show pixel boundaries
    for i in range(input_2d.shape[0] + 1):
        ax.axhline(i - 0.5, color='red', linewidth=0.5, alpha=0.3)
    for j in range(input_2d.shape[1] + 1):
        ax.axvline(j - 0.5, color='red', linewidth=0.5, alpha=0.3)
    
    # Plot 2: Kernel/Filter
    ax = axes[0, 1]
    im2 = ax.imshow(kernel_2d, cmap='RdBu_r')
    ax.set_title('Convolution Kernel')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im2, ax=ax)
    
    # Add text annotations showing kernel values
    for i in range(kernel_2d.shape[0]):
        for j in range(kernel_2d.shape[1]):
            ax.text(j, i, f'{kernel_2d[i, j]:.2f}', 
                   ha='center', va='center', color='black', fontsize=10)
    
    # Plot 3: Example convolution at position (0, 0)
    ax = axes[1, 0]
    K = kernel_2d.shape[0]
    
    # Extract window at first position
    if padding > 0:
        padded_input = np.pad(
            input_2d,
            ((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_input = input_2d
    
    window = padded_input[:K, :K]
    element_wise = window * kernel_2d
    
    im3 = ax.imshow(element_wise, cmap='RdBu_r')
    ax.set_title(f'Element-wise Multiplication at Position (0,0)\nSum = {element_wise.sum():.2f}')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im3, ax=ax)
    
    # Add text annotations
    for i in range(element_wise.shape[0]):
        for j in range(element_wise.shape[1]):
            ax.text(j, i, f'{element_wise[i, j]:.2f}', 
                   ha='center', va='center', color='black', fontsize=9)
    
    # Plot 4: Output feature map
    ax = axes[1, 1]
    im4 = ax.imshow(output, cmap='viridis')
    ax.set_title('Output Feature Map')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im4, ax=ax)
    
    # Add text showing dimensions
    fig.text(0.5, 0.02, 
            f'Input: {input_2d.shape} | Kernel: {kernel_2d.shape} | Stride: {stride} | Padding: {padding} | Output: {output.shape}',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
