"""Backpropagation Training Implementation.

Extends the MLP class with training capabilities using backpropagation
and gradient descent.

HYPERPARAMETER GUIDANCE:
- See HYPERPARAMETER_INSIGHTS.md for detailed guidance on:
  * Learning rate selection (recommended: 0.3-0.5)
  * Network architecture choices (2-2-1 minimal, 2-4-1 robust)
  * Random seed effects and initialization
  * Epoch requirements and convergence patterns
  * Troubleshooting common training issues

QUICK START:
    # Minimal configuration (educational)
    mlp = TrainableMLP([2, 2, 1], ['sigmoid', 'sigmoid'], 
                       learning_rate=0.3, random_state=123)
    
    # Robust configuration (recommended)
    mlp = TrainableMLP([2, 4, 1], ['sigmoid', 'sigmoid'], 
                       learning_rate=0.5, random_state=123)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import sys
import os

# Add parent directory to path to import MLP
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02-multi-layer-perceptron'))
from mlp import MLP, sigmoid, tanh, relu, get_activation


# ============================================================================
# Activation Derivative Functions
# ============================================================================

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation function.
    
    Formula: σ'(z) = σ(z) * (1 - σ(z))
    
    This derivative is used during backpropagation to compute gradients.
    The sigmoid function squashes values to (0, 1), and its derivative
    is largest at z=0 (where σ(z)=0.5) and approaches 0 for large |z|.
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Derivative values with same shape as input
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of hyperbolic tangent activation function.
    
    Formula: tanh'(z) = 1 - tanh²(z)
    
    This derivative is used during backpropagation to compute gradients.
    The tanh function squashes values to (-1, 1), and its derivative
    is largest at z=0 (where tanh(z)=0) and approaches 0 for large |z|.
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Derivative values with same shape as input
    """
    t = tanh(z)
    return 1 - t ** 2


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation function.
    
    Formula: ReLU'(z) = 1 if z > 0, else 0
    
    This derivative is used during backpropagation to compute gradients.
    ReLU has a simple derivative: 1 for positive inputs, 0 otherwise.
    This simplicity makes it computationally efficient and helps avoid
    vanishing gradients.
    
    Args:
        z: Input array (pre-activation values)
        
    Returns:
        Derivative values (0 or 1) with same shape as input
    """
    return (z > 0).astype(float)


def get_activation_derivative(name: str) -> Callable:
    """
    Get activation derivative function by name.
    
    Args:
        name: Activation function name ('sigmoid', 'tanh', or 'relu')
        
    Returns:
        Activation derivative function
        
    Raises:
        ValueError: If activation name is not recognized
    """
    derivatives = {
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'relu': relu_derivative
    }
    
    if name not in derivatives:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Supported: {list(derivatives.keys())}"
        )
    
    return derivatives[name]


# ============================================================================
# TrainableMLP Class
# ============================================================================

class TrainableMLP(MLP):
    """
    MLP with backpropagation training capability.
    
    Extends the base MLP class with methods for training via gradient descent.
    This implementation demonstrates how neural networks learn automatically
    by computing gradients through backpropagation and updating weights to
    minimize a loss function.
    
    Attributes:
        learning_rate: Step size for gradient descent weight updates
        (inherits all MLP attributes: layer_sizes, activations, weights_, biases_)
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        learning_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize trainable MLP.
        
        Args:
            layer_sizes: Number of neurons in each layer
                        Example: [2, 4, 1] = 2 inputs, 4 hidden, 1 output
            activations: Activation function for each layer transition
            learning_rate: Step size for gradient descent (default: 0.1)
            random_state: Random seed for weight initialization
            
        Raises:
            ValueError: If learning_rate is not positive
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Initialize parent MLP class
        super().__init__(layer_sizes, activations, random_state)
        
        self.learning_rate = learning_rate
    
    def _forward_with_cache(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Compute forward pass and cache pre-activation values for backprop.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            output: Final layer output
            activations: List of activation values [input, hidden1, ..., output]
            pre_activations: List of pre-activation values [z1, z2, ...]
        """
        # Validate input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input has {X.shape[1]} features but network expects "
                f"{self.layer_sizes[0]} features"
            )
        
        # Store activations and pre-activations
        layer_activations = [X]
        pre_activations = []
        current_input = X
        
        # Forward pass through each layer
        for i in range(len(self.weights_)):
            # Linear transformation: z = X @ W + b
            z = current_input @ self.weights_[i] + self.biases_[i]
            pre_activations.append(z)
            
            # Apply activation function
            activation_fn = get_activation(self.activations[i])
            a = activation_fn(z)
            
            layer_activations.append(a)
            current_input = a
        
        return current_input, layer_activations, pre_activations
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.
        
        MSE measures the average squared difference between predictions
        and true values. Lower MSE means better predictions.
        
        Formula: MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true: True target values of shape (n_samples, n_outputs)
            y_pred: Predicted values of shape (n_samples, n_outputs)
            
        Returns:
            Mean squared error as a scalar float
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer_activations: List[np.ndarray],
        pre_activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute gradients via backpropagation.
        
        This implements the backward pass of backpropagation, computing
        gradients of the loss with respect to all weights and biases using
        the chain rule from calculus.
        
        The algorithm works backward from the output layer to the input layer:
        1. Compute output layer error (derivative of loss)
        2. For each layer (from output to input):
           - Compute gradient with respect to weights and biases
           - Propagate error backward to previous layer using chain rule
        
        Args:
            X: Input data of shape (n_samples, n_features)
            y: True target values of shape (n_samples, n_outputs)
            layer_activations: List of activation values from forward pass
                              [input, hidden1, hidden2, ..., output]
            pre_activations: List of pre-activation values [z1, z2, ...]
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
            - weight_gradients: List of gradient arrays for each weight matrix
            - bias_gradients: List of gradient arrays for each bias vector
        """
        n_samples = X.shape[0]
        n_layers = len(self.weights_)
        
        # Initialize gradient storage
        weight_gradients = [None] * n_layers
        bias_gradients = [None] * n_layers
        
        # Backpropagate through each layer (from output to input)
        # Start with output layer error
        # For MSE loss: dL/da = 2(a - y) where a is the output activation
        delta = layer_activations[-1] - y
        
        for i in range(n_layers - 1, -1, -1):
            # Current layer's input (previous layer's output)
            layer_input = layer_activations[i]
            
            # Apply activation derivative: delta_z = delta_a * activation'(z)
            activation_derivative = get_activation_derivative(self.activations[i])
            delta_z = delta * activation_derivative(pre_activations[i])
            
            # Compute gradients for weights and biases at this layer
            # dL/dW = (1/n) * layer_input.T @ delta_z
            # dL/db = (1/n) * sum(delta_z, axis=0)
            weight_gradients[i] = (layer_input.T @ delta_z) / n_samples
            bias_gradients[i] = np.mean(delta_z, axis=0)
            
            # Propagate error to previous layer (if not at input layer)
            if i > 0:
                # delta for previous layer's activation: delta_a_prev = delta_z @ W.T
                delta = delta_z @ self.weights_[i].T
        
        return weight_gradients, bias_gradients

    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train the network using backpropagation and gradient descent.
        
        This method implements the complete training loop:
        1. Forward pass: Compute predictions
        2. Compute loss: Measure how wrong predictions are
        3. Backward pass: Compute gradients via backpropagation
        4. Update weights: Apply gradient descent to minimize loss
        5. Track progress: Record loss history
        
        The network learns by repeatedly adjusting weights in the direction
        that reduces the loss function (gradient descent).
        
        Args:
            X: Training inputs of shape (n_samples, n_features)
            y: Training targets of shape (n_samples, n_outputs)
            epochs: Number of training iterations (default: 1000)
            verbose: Whether to print training progress (default: False)
            
        Returns:
            Dictionary with training history:
                - 'loss': List of loss values for each epoch
                
        Raises:
            ValueError: If X and y have mismatched sample counts or invalid shapes
        """
        # Validate inputs
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        
        # Initialize loss tracking
        loss_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass: Compute predictions and store activations
            y_pred, layer_activations, pre_activations = self._forward_with_cache(X)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            loss_history.append(loss)
            
            # Backward pass: Compute gradients
            weight_gradients, bias_gradients = self._backward(X, y, layer_activations, pre_activations)
            
            # Update weights using gradient descent
            # Formula: w = w - learning_rate * gradient
            for i in range(len(self.weights_)):
                self.weights_[i] -= self.learning_rate * weight_gradients[i]
                self.biases_[i] -= self.learning_rate * bias_gradients[i]
            
            # Print progress if verbose
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss:.6f}")
        
        # Return training history
        return {'loss': loss_history}
