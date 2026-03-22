"""
Complete CNN architecture combining convolution, pooling, and fully-connected layers.

This module implements:
- CNN class: Complete CNN architecture with configurable layers
- create_simple_cnn(): Pre-configured CNN for MNIST-like tasks
- create_alexnet_inspired(): Deeper CNN inspired by AlexNet

All implementations use NumPy only for educational clarity.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from convolution import convolve_2d
from pooling import max_pool_2d

# Type aliases
NDArray = np.ndarray
ConvConfig = Dict
PoolConfig = Dict


def relu(z: NDArray) -> NDArray:
    """ReLU activation function: max(0, z)."""
    return np.maximum(0, z)


def softmax(z: NDArray) -> NDArray:
    """Softmax activation function for output layer."""
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class CNN:
    """
    Complete CNN architecture for image classification.
    
    This class combines convolution, pooling, and fully-connected layers
    to build a complete neural network for image classification tasks.
    
    Attributes:
        input_shape: Shape of input images (H, W, C)
        conv_layers: List of convolution layer configurations
        pool_layers: List of pooling layer configurations
        fc_layers: List of fully-connected layer sizes
        num_classes: Number of output classes
        params: Dictionary of learnable parameters
    
    Example:
        >>> cnn = create_simple_cnn(input_shape=(28, 28, 1), num_classes=10)
        >>> X = np.random.randn(4, 28, 28, 1)
        >>> output = cnn.forward(X)
        >>> output.shape
        (4, 10)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        conv_configs: List[ConvConfig],
        pool_configs: List[PoolConfig],
        fc_sizes: List[int],
        num_classes: int = 10,
        seed: int = 42
    ):
        """
        Initialize CNN with specified architecture.
        
        Args:
            input_shape: Shape of input images (H, W, C)
            conv_configs: List of convolution layer configurations
            pool_configs: List of pooling layer configurations
            fc_sizes: List of fully-connected layer sizes
            num_classes: Number of output classes (default: 10)
            seed: Random seed for reproducibility (default: 42)
        
        Raises:
            ValueError: If architecture configuration is invalid
        """
        np.random.seed(seed)
        
        # Validate inputs
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (H, W, C), got {input_shape}")
        
        if not conv_configs:
            raise ValueError("conv_configs cannot be empty")
        
        if not pool_configs:
            raise ValueError("pool_configs cannot be empty")
        
        if len(conv_configs) != len(pool_configs):
            raise ValueError(
                f"Number of conv layers ({len(conv_configs)}) must match "
                f"number of pool layers ({len(pool_configs)})"
            )
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.input_shape = input_shape
        self.conv_configs = conv_configs
        self.pool_configs = pool_configs
        self.fc_sizes = fc_sizes
        self.num_classes = num_classes
        self.params = {}
        
        # Initialize convolution layer parameters
        self._initialize_conv_params()
        
        # Calculate flattened size after conv and pooling
        self.flattened_size = self._calculate_flattened_size()
        
        # Initialize fully-connected layer parameters
        self._initialize_fc_params()
    
    def _initialize_conv_params(self) -> None:
        """Initialize convolution layer parameters."""
        H, W, C_in = self.input_shape
        
        for layer_idx, config in enumerate(self.conv_configs):
            num_filters = config['num_filters']
            kernel_size = config['kernel_size']
            
            # He initialization for ReLU
            scale = np.sqrt(2.0 / (kernel_size * kernel_size * C_in))
            
            # Initialize kernel: (kernel_size, kernel_size, C_in, num_filters)
            kernel = np.random.randn(
                kernel_size, kernel_size, C_in, num_filters
            ) * scale
            
            # Initialize bias
            bias = np.zeros(num_filters)
            
            self.params[f'conv_{layer_idx}_kernel'] = kernel
            self.params[f'conv_{layer_idx}_bias'] = bias
            
            # Update input channels for next layer
            C_in = num_filters
    
    def _calculate_flattened_size(self) -> int:
        """Calculate the size of flattened feature maps after conv and pooling."""
        H, W, C = self.input_shape
        
        for layer_idx, (conv_config, pool_config) in enumerate(
            zip(self.conv_configs, self.pool_configs)
        ):
            # After convolution
            kernel_size = conv_config['kernel_size']
            stride = conv_config.get('stride', 1)
            padding = conv_config.get('padding', 0)
            
            H = (H - kernel_size + 2 * padding) // stride + 1
            W = (W - kernel_size + 2 * padding) // stride + 1
            C = conv_config['num_filters']
            
            # After pooling
            pool_size = pool_config['pool_size']
            pool_stride = pool_config.get('stride', pool_size)
            
            H = (H - pool_size) // pool_stride + 1
            W = (W - pool_size) // pool_stride + 1
        
        return H * W * C
    
    def _initialize_fc_params(self) -> None:
        """Initialize fully-connected layer parameters."""
        # First FC layer: from flattened features
        layer_sizes = [self.flattened_size] + self.fc_sizes + [self.num_classes]
        
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (n_in + n_out))
            W = np.random.randn(n_in, n_out) * scale
            b = np.zeros(n_out)
            
            self.params[f'fc_{i}_W'] = W
            self.params[f'fc_{i}_b'] = b
    
    def forward(self, X: NDArray) -> NDArray:
        """
        Forward pass through all layers.
        
        Applies convolution layers with ReLU activation, pooling layers,
        flattening, fully-connected layers with ReLU, and softmax.
        
        Args:
            X: Input batch of shape (batch_size, H, W, C)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        
        Raises:
            ValueError: If input shape doesn't match expected shape
        
        Example:
            >>> cnn = create_simple_cnn()
            >>> X = np.random.randn(4, 28, 28, 1)
            >>> logits = cnn.forward(X)
            >>> logits.shape
            (4, 10)
        """
        batch_size = X.shape[0]
        
        # Validate input shape
        if X.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input shape {X.shape[1:]} doesn't match expected {self.input_shape}"
            )
        
        # Apply convolution and pooling layers
        current = X
        for layer_idx, (conv_config, pool_config) in enumerate(
            zip(self.conv_configs, self.pool_configs)
        ):
            # Apply convolution to each sample in batch
            conv_outputs = []
            for sample_idx in range(batch_size):
                sample = current[sample_idx]  # Shape: (H, W, C)
                
                # Apply each filter
                num_filters = conv_config['num_filters']
                kernel_size = conv_config['kernel_size']
                stride = conv_config.get('stride', 1)
                padding = conv_config.get('padding', 0)
                
                kernel = self.params[f'conv_{layer_idx}_kernel']
                bias = self.params[f'conv_{layer_idx}_bias']
                
                # Convolve with each filter
                feature_maps = []
                for f_idx in range(num_filters):
                    # Extract filter for this output channel
                    filter_kernel = kernel[:, :, :, f_idx]
                    
                    # Apply convolution
                    feature_map = convolve_2d(
                        sample, filter_kernel, stride=stride, padding=padding
                    )
                    
                    # Add bias and apply ReLU
                    feature_map = relu(feature_map + bias[f_idx])
                    feature_maps.append(feature_map)
                
                # Stack feature maps: (H_out, W_out, num_filters)
                sample_conv = np.stack(feature_maps, axis=2)
                conv_outputs.append(sample_conv)
            
            # Stack batch: (batch_size, H_out, W_out, num_filters)
            current = np.stack(conv_outputs, axis=0)
            
            # Apply pooling
            pool_size = pool_config['pool_size']
            pool_stride = pool_config.get('stride', pool_size)
            
            pooled_outputs = []
            for sample_idx in range(batch_size):
                pooled = max_pool_2d(
                    current[sample_idx], pool_size=pool_size, stride=pool_stride
                )
                pooled_outputs.append(pooled)
            
            current = np.stack(pooled_outputs, axis=0)
        
        # Flatten feature maps: (batch_size, flattened_size)
        current = current.reshape(batch_size, -1)
        
        # Apply fully-connected layers
        num_fc_layers = len(self.fc_sizes) + 1  # +1 for output layer
        for fc_idx in range(num_fc_layers):
            W = self.params[f'fc_{fc_idx}_W']
            b = self.params[f'fc_{fc_idx}_b']
            
            # Linear transformation
            current = current @ W + b
            
            # Apply ReLU to all but last layer
            if fc_idx < num_fc_layers - 1:
                current = relu(current)
        
        # Apply softmax to output
        output = softmax(current)
        
        return output
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Make predictions (inference mode).
        
        Calls forward pass to get logits and returns argmax (class with
        highest probability).
        
        Args:
            X: Input batch of shape (batch_size, H, W, C)
        
        Returns:
            Predicted class indices of shape (batch_size,)
        
        Example:
            >>> cnn = create_simple_cnn()
            >>> X = np.random.randn(4, 28, 28, 1)
            >>> predictions = cnn.predict(X)
            >>> predictions.shape
            (4,)
            >>> np.all((predictions >= 0) & (predictions < 10))
            True
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
    
    def count_parameters(self) -> int:
        """
        Calculate total number of learnable parameters.
        
        Sums parameters from all convolution and fully-connected layers.
        Pooling layers have no learnable parameters.
        
        Returns:
            Total number of parameters
        
        Example:
            >>> cnn = create_simple_cnn()
            >>> params = cnn.count_parameters()
            >>> params > 0
            True
        """
        total = 0
        for param_name, param_value in self.params.items():
            total += param_value.size
        return total
    
    def get_layer_outputs(self, X: NDArray) -> Dict[str, NDArray]:
        """
        Get intermediate outputs from each layer.
        
        Performs forward pass and stores outputs from each layer.
        Useful for feature visualization and debugging.
        
        Args:
            X: Input batch of shape (batch_size, H, W, C)
        
        Returns:
            Dictionary mapping layer names to their outputs
        
        Example:
            >>> cnn = create_simple_cnn()
            >>> X = np.random.randn(4, 28, 28, 1)
            >>> outputs = cnn.get_layer_outputs(X)
            >>> 'conv_0' in outputs
            True
        """
        batch_size = X.shape[0]
        layer_outputs = {}
        
        # Store input
        layer_outputs['input'] = X
        
        # Apply convolution and pooling layers
        current = X
        for layer_idx, (conv_config, pool_config) in enumerate(
            zip(self.conv_configs, self.pool_configs)
        ):
            # Apply convolution to each sample in batch
            conv_outputs = []
            for sample_idx in range(batch_size):
                sample = current[sample_idx]
                
                num_filters = conv_config['num_filters']
                kernel_size = conv_config['kernel_size']
                stride = conv_config.get('stride', 1)
                padding = conv_config.get('padding', 0)
                
                kernel = self.params[f'conv_{layer_idx}_kernel']
                bias = self.params[f'conv_{layer_idx}_bias']
                
                feature_maps = []
                for f_idx in range(num_filters):
                    filter_kernel = kernel[:, :, :, f_idx]
                    feature_map = convolve_2d(
                        sample, filter_kernel, stride=stride, padding=padding
                    )
                    feature_map = relu(feature_map + bias[f_idx])
                    feature_maps.append(feature_map)
                
                sample_conv = np.stack(feature_maps, axis=2)
                conv_outputs.append(sample_conv)
            
            current = np.stack(conv_outputs, axis=0)
            layer_outputs[f'conv_{layer_idx}'] = current.copy()
            
            # Apply pooling
            pool_size = pool_config['pool_size']
            pool_stride = pool_config.get('stride', pool_size)
            
            pooled_outputs = []
            for sample_idx in range(batch_size):
                pooled = max_pool_2d(
                    current[sample_idx], pool_size=pool_size, stride=pool_stride
                )
                pooled_outputs.append(pooled)
            
            current = np.stack(pooled_outputs, axis=0)
            layer_outputs[f'pool_{layer_idx}'] = current.copy()
        
        # Flatten
        current = current.reshape(batch_size, -1)
        layer_outputs['flatten'] = current.copy()
        
        # Apply fully-connected layers
        num_fc_layers = len(self.fc_sizes) + 1
        for fc_idx in range(num_fc_layers):
            W = self.params[f'fc_{fc_idx}_W']
            b = self.params[f'fc_{fc_idx}_b']
            
            current = current @ W + b
            
            if fc_idx < num_fc_layers - 1:
                current = relu(current)
                layer_outputs[f'fc_{fc_idx}'] = current.copy()
            else:
                # Output layer (before softmax)
                layer_outputs[f'fc_{fc_idx}_logits'] = current.copy()
        
        # Apply softmax
        current = softmax(current)
        layer_outputs['output'] = current
        
        return layer_outputs


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    seed: int = 42
) -> CNN:
    """
    Create a simple CNN for MNIST-like tasks.
    
    Architecture:
    - Conv(32 filters, 3×3) → ReLU → MaxPool(2×2)
    - Conv(64 filters, 3×3) → ReLU → MaxPool(2×2)
    - Flatten → FC(128) → ReLU → FC(num_classes)
    
    Total parameters: ~100K
    
    Args:
        input_shape: Shape of input images (default: (28, 28, 1))
        num_classes: Number of output classes (default: 10)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Initialized CNN instance
    
    Example:
        >>> cnn = create_simple_cnn()
        >>> X = np.random.randn(4, 28, 28, 1)
        >>> output = cnn.forward(X)
        >>> output.shape
        (4, 10)
    """
    conv_configs = [
        {
            'num_filters': 32,
            'kernel_size': 3,
            'stride': 1,
            'padding': 0
        },
        {
            'num_filters': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 0
        }
    ]
    
    pool_configs = [
        {
            'pool_size': 2,
            'stride': 2
        },
        {
            'pool_size': 2,
            'stride': 2
        }
    ]
    
    fc_sizes = [128]
    
    return CNN(
        input_shape=input_shape,
        conv_configs=conv_configs,
        pool_configs=pool_configs,
        fc_sizes=fc_sizes,
        num_classes=num_classes,
        seed=seed
    )


def create_alexnet_inspired(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    seed: int = 42
) -> CNN:
    """
    Create a deeper CNN inspired by AlexNet.
    
    Architecture:
    - Conv(96 filters, 11×11, stride=4) → ReLU → MaxPool(3×3)
    - Conv(256 filters, 5×5) → ReLU → MaxPool(3×3)
    - Conv(384 filters, 3×3) → ReLU
    - Conv(384 filters, 3×3) → ReLU
    - Conv(256 filters, 3×3) → ReLU → MaxPool(3×3)
    - Flatten → FC(4096) → ReLU → FC(4096) → ReLU → FC(num_classes)
    
    Total parameters: ~60M (simplified version)
    
    Args:
        input_shape: Shape of input images (default: (224, 224, 3))
        num_classes: Number of output classes (default: 10)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Initialized CNN instance
    
    Example:
        >>> cnn = create_alexnet_inspired()
        >>> X = np.random.randn(2, 224, 224, 3)
        >>> output = cnn.forward(X)
        >>> output.shape
        (2, 10)
    """
    conv_configs = [
        {
            'num_filters': 96,
            'kernel_size': 11,
            'stride': 4,
            'padding': 0
        },
        {
            'num_filters': 256,
            'kernel_size': 5,
            'stride': 1,
            'padding': 2
        },
        {
            'num_filters': 384,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        {
            'num_filters': 384,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        {
            'num_filters': 256,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
    ]
    
    pool_configs = [
        {
            'pool_size': 3,
            'stride': 2
        },
        {
            'pool_size': 3,
            'stride': 2
        },
        {
            'pool_size': 1,
            'stride': 1
        },
        {
            'pool_size': 1,
            'stride': 1
        },
        {
            'pool_size': 3,
            'stride': 2
        }
    ]
    
    fc_sizes = [4096, 4096]
    
    return CNN(
        input_shape=input_shape,
        conv_configs=conv_configs,
        pool_configs=pool_configs,
        fc_sizes=fc_sizes,
        num_classes=num_classes,
        seed=seed
    )
