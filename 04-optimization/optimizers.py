# Optimizer implementations for blog post #4

import numpy as np


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, learning_rate: float):
        """Initialize optimizer with learning rate.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params: dict, gradients: dict) -> dict:
        """Update parameters given gradients.
        
        Args:
            params: Dictionary of parameters to update
            gradients: Dictionary of gradients for each parameter
            
        Returns:
            Dictionary of updated parameters
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def reset(self):
        """Reset internal state (for momentum, moving averages, etc.)"""
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    Same as VanillaGD, but called with mini-batch gradients.
    The "stochastic" aspect comes from using mini-batches in the training loop.
    """
    
    def update(self, params: dict, gradients: dict) -> dict:
        """Update parameters using stochastic gradient descent.
        
        Args:
            params: Dictionary of parameters to update
            gradients: Dictionary of gradients for each parameter
            
        Returns:
            Dictionary of updated parameters
        """
        updated_params = {}
        for key in params:
            updated_params[key] = params[key] - self.learning_rate * gradients[key]
        return updated_params


class MomentumSGD(Optimizer):
    """SGD with Momentum optimizer.
    
    Maintains velocity state: v = momentum * v - lr * grad
    Update weights: w = w + v
    """
    
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        """Initialize Momentum SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient (default 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: dict, gradients: dict) -> dict:
        """Update parameters using momentum.
        
        Args:
            params: Dictionary of parameters to update
            gradients: Dictionary of gradients for each parameter
            
        Returns:
            Dictionary of updated parameters
        """
        if not self.velocity:
            # Initialize velocity on first call
            self.velocity = {key: np.zeros_like(params[key]) for key in params}
        
        updated_params = {}
        for key in params:
            # v = momentum * v - lr * grad
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * gradients[key]
            # w = w + v
            updated_params[key] = params[key] + self.velocity[key]
        return updated_params
    
    def reset(self):
        """Reset velocity state."""
        self.velocity = {}


class Adam(Optimizer):
    """Adam optimizer.
    
    Maintains first moment (m) and second moment (v).
    Applies bias correction.
    Adaptive learning rate per parameter.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates (default 0.001)
            beta1: Exponential decay rate for first moment (default 0.9)
            beta2: Exponential decay rate for second moment (default 0.999)
            epsilon: Small constant for numerical stability (default 1e-8)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def update(self, params: dict, gradients: dict) -> dict:
        """Update parameters using Adam.
        
        Args:
            params: Dictionary of parameters to update
            gradients: Dictionary of gradients for each parameter
            
        Returns:
            Dictionary of updated parameters
        """
        if not self.m:
            # Initialize moments on first call
            self.m = {key: np.zeros_like(params[key]) for key in params}
            self.v = {key: np.zeros_like(params[key]) for key in params}
        
        self.t += 1
        updated_params = {}
        
        for key in params:
            # Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            
            # Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def reset(self):
        """Reset moment estimates and time step."""
        self.m = {}
        self.v = {}
        self.t = 0
