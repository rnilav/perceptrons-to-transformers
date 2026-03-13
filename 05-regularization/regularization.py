# Dropout and weight decay implementations for regularization

import numpy as np


class Dropout:
    """Dropout layer that randomly deactivates neurons during training."""
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize Dropout layer.
        
        Args:
            dropout_rate: Probability of deactivating each neuron (0.0 to 1.0)
        """
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training: bool = True):
        """
        Apply dropout during training, pass through during inference.
        
        During training: randomly deactivate neurons and scale activations.
        During inference: pass through unchanged (all neurons active).
        
        Args:
            X: Input activations (batch_size, num_neurons)
            training: Whether in training mode (dropout active) or inference mode
        
        Returns:
            X with dropout applied (if training), or X unchanged (if inference)
        """
        if not training or self.dropout_rate == 0:
            self.mask = None
            return X
        
        # Generate random mask: 1 with probability (1 - dropout_rate)
        # This means neurons are kept with probability (1 - dropout_rate)
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, X.shape)
        
        # Scale activations to maintain expected value
        # Divide by (1 - dropout_rate) to keep expected value same
        # This is called "inverted dropout"
        return X * self.mask / (1 - self.dropout_rate)
    
    def backward(self, dX):
        """
        Backpropagate through dropout layer.
        
        Apply the same mask used in forward pass to gradients,
        and scale by the same factor.
        
        Args:
            dX: Gradient of loss with respect to output
        
        Returns:
            Gradient of loss with respect to input
        """
        if self.mask is None:
            return dX
        
        # Apply same mask to gradients and scale appropriately
        return dX * self.mask / (1 - self.dropout_rate)


def compute_weight_decay_penalty(params: dict, weight_decay: float) -> float:
    """
    Compute L2 regularization penalty.
    
    The penalty is computed as: λ * sum(W²) for all weights.
    Only weights are penalized, not biases.
    
    Args:
        params: Dictionary of network parameters (weights and biases)
                Expected keys: 'W1', 'b1', 'W2', 'b2', etc.
        weight_decay: Coefficient λ for regularization (typically 0.0001 to 0.01)
    
    Returns:
        Penalty term: λ * sum(w²) for all weights
    """
    penalty = 0.0
    for key in params:
        if 'W' in key:  # Only penalize weights, not biases
            penalty += np.sum(params[key] ** 2)
    
    return weight_decay * penalty


def apply_weight_decay_to_gradients(gradients: dict, params: dict, weight_decay: float) -> dict:
    """
    Add weight decay term to gradients.
    
    This is equivalent to adding λ * w to the gradient for each weight.
    The effect is to push weights toward zero during optimization.
    
    Args:
        gradients: Dictionary of computed gradients from backpropagation
                   Expected keys: 'W1', 'b1', 'W2', 'b2', etc.
        params: Dictionary of current parameters (weights and biases)
        weight_decay: Coefficient λ for regularization
    
    Returns:
        Modified gradients with weight decay applied to weights only
    """
    updated_gradients = gradients.copy()
    
    for key in params:
        if 'W' in key:  # Only apply to weights, not biases
            updated_gradients[key] = gradients[key] + weight_decay * params[key]
    
    return updated_gradients
