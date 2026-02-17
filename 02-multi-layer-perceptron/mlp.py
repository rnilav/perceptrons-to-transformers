"""Multi-Layer Perceptron Implementation.

A minimal, educational implementation of MLPs built from scratch.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


# ============================================================================
# Activation Functions
# ============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Formula: σ(z) = 1 / (1 + e^(-z))
    Range: (0, 1)
    
    Args:
        z: Input array
        
    Returns:
        Activated output in range (0, 1)
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    
    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Range: (-1, 1)
    
    Args:
        z: Input array
        
    Returns:
        Activated output in range (-1, 1)
    """
    return np.tanh(z)


def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation function.
    
    Formula: ReLU(z) = max(0, z)
    Range: [0, ∞)
    
    Args:
        z: Input array
        
    Returns:
        Activated output (negative values become 0)
    """
    return np.maximum(0, z)


def get_activation(name: str) -> Callable:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name ('sigmoid', 'tanh', or 'relu')
        
    Returns:
        Activation function
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu
    }
    
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Supported: {list(activations.keys())}"
        )
    
    return activations[name]


# ============================================================================
# MLP Class
# ============================================================================

class MLP:
    """
    Multi-Layer Perceptron for educational purposes.
    
    A neural network with one or more hidden layers that can learn
    non-linear decision boundaries. This implementation focuses on
    clarity and understanding over optimization.
    
    Attributes:
        layer_sizes: List of integers specifying neurons per layer
        activations: List of activation function names
        weights_: List of weight matrices (set after initialization)
        biases_: List of bias vectors (set after initialization)
        random_state: Seed for reproducibility
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        random_state: Optional[int] = None
    ):
        """
        Initialize MLP architecture.
        
        Args:
            layer_sizes: Number of neurons in each layer
                        Example: [2, 4, 1] = 2 inputs, 4 hidden, 1 output
            activations: Activation function for each layer transition
                        Length must be len(layer_sizes) - 1
            random_state: Random seed for weight initialization
            
        Raises:
            ValueError: If layer_sizes or activations are invalid
        """
        # Validate inputs
        if not layer_sizes or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 layers (input and output)")
        
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers")
        
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"activations length ({len(activations)}) must equal "
                f"number of layer transitions ({len(layer_sizes) - 1})"
            )
        
        # Validate activation names
        for act in activations:
            get_activation(act)  # Will raise ValueError if invalid
        
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.random_state = random_state
        
        # Initialize weights and biases
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights and biases with small random values."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights_ = []
        self.biases_ = []
        
        # Create weight matrix and bias vector for each layer transition
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # Xavier/Glorot initialization for better convergence
            scale = np.sqrt(2.0 / (n_in + n_out))
            W = np.random.randn(n_in, n_out) * scale
            b = np.zeros(n_out)
            
            self.weights_.append(W)
            self.biases_.append(b)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            output: Final layer output of shape (n_samples, n_outputs)
            activations: List of activation values for each layer
            
        Raises:
            ValueError: If input shape doesn't match network input size
        """
        # Validate input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input has {X.shape[1]} features but network expects "
                f"{self.layer_sizes[0]} features"
            )
        
        # Store activations for each layer
        layer_activations = [X]
        current_input = X
        
        # Forward pass through each layer
        for i in range(len(self.weights_)):
            # Linear transformation: z = X @ W + b
            z = current_input @ self.weights_[i] + self.biases_[i]
            
            # Apply activation function
            activation_fn = get_activation(self.activations[i])
            a = activation_fn(z)
            
            layer_activations.append(a)
            current_input = a
        
        return current_input, layer_activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (forward pass without intermediate values).
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        output, _ = self.forward(X)
        return output
    
    def summary(self) -> str:
        """
        Return a string summary of network architecture.
        
        Returns:
            Multi-line string describing the network
        """
        lines = ["=" * 60]
        lines.append("MLP Architecture Summary")
        lines.append("=" * 60)
        
        total_params = 0
        
        for i in range(len(self.layer_sizes) - 1):
            layer_name = "Input" if i == 0 else f"Hidden {i}"
            next_layer = "Output" if i == len(self.layer_sizes) - 2 else f"Hidden {i+1}"
            
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            activation = self.activations[i]
            
            # Calculate parameters: weights + biases
            params = n_in * n_out + n_out
            total_params += params
            
            lines.append(f"\nLayer {i+1}: {layer_name} → {next_layer}")
            lines.append(f"  Shape: ({n_in}, {n_out})")
            lines.append(f"  Activation: {activation}")
            lines.append(f"  Parameters: {params:,}")
        
        lines.append("\n" + "=" * 60)
        lines.append(f"Total Parameters: {total_params:,}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

def create_xor_network() -> MLP:
    """
    Create a pre-configured 2-2-1 network that solves XOR.
    
    This network uses hand-crafted weights that demonstrate how
    hidden layers enable non-linear classification.
    
    Architecture:
        - 2 inputs (x1, x2)
        - 2 hidden neurons with sigmoid activation
        - 1 output neuron with sigmoid activation
    
    How it works:
        - Hidden neuron 1: Learns OR-like pattern (x1 OR x2)
        - Hidden neuron 2: Learns AND-like pattern (x1 AND x2)
        - Output: Combines to produce XOR (OR AND NOT AND)
    
    Returns:
        MLP configured to solve XOR problem
    """
    mlp = MLP(
        layer_sizes=[2, 2, 1],
        activations=['sigmoid', 'sigmoid'],
        random_state=42
    )
    
    # Hand-crafted weights for XOR solution
    # Using realistic weight values (2-4 range) for smooth boundaries
    
    # Layer 1: Input → Hidden
    # Hidden neuron 1 detects when either input is 1 (OR-like)
    # Hidden neuron 2 detects when both inputs are 1 (AND-like)
    mlp.weights_[0] = np.array([
        [3.5, 3.5],    # Weights from input 1
        [3.5, 3.5]     # Weights from input 2
    ])
    mlp.biases_[0] = np.array([-1.5, -5.0])
    
    # Layer 2: Hidden → Output
    # Output combines: "OR but not AND" = XOR
    mlp.weights_[1] = np.array([
        [4.0],     # Weight from hidden neuron 1 (OR)
        [-4.0]     # Weight from hidden neuron 2 (AND)
    ])
    mlp.biases_[1] = np.array([-2.0])
    
    return mlp
