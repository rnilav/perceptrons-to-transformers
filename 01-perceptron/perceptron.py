"""
Perceptron implementation for educational purposes.

This module provides a from-scratch implementation of the perceptron algorithm,
the foundation of neural networks.
"""

import numpy as np
from typing import Optional, List


class Perceptron:
    """
    Single-layer perceptron for binary classification.
    
    The perceptron is a linear classifier that learns a decision boundary
    by updating weights based on misclassified examples. It is guaranteed
    to converge if the data is linearly separable.
    
    Attributes:
        learning_rate: Step size for weight updates (α)
        n_iterations: Maximum number of training epochs
        random_state: Seed for random number generator
        weights_: Learned weight vector (shape: n_features + 1)
                  First element is bias, rest are feature weights
        errors_: Number of misclassifications in each epoch
        
    Mathematical Formulation:
        ŷ = σ(w^T x + b)
        
        where σ is the step function:
        σ(z) = 1 if z ≥ 0, else 0
        
    Example:
        >>> from perceptron import Perceptron
        >>> import numpy as np
        >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> y = np.array([0, 0, 0, 1])  # AND function
        >>> model = Perceptron(learning_rate=0.1, n_iterations=10)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> print(predictions)
        [0 0 0 1]
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize the perceptron.
        
        Args:
            learning_rate: Learning rate α for weight updates (0 < α ≤ 1)
            n_iterations: Maximum number of training epochs
            random_state: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None
        self.errors_: List[int] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Train the perceptron on training data.
        
        Uses the perceptron learning rule:
        w ← w + α * (y - ŷ) * x
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,), values in {0, 1}
            
        Returns:
            self: Fitted perceptron instance
            
        Raises:
            ValueError: If X and y have incompatible shapes
            
        Notes:
            - Weights are initialized to small random values
            - Training stops early if no errors occur in an epoch
            - The algorithm is guaranteed to converge if data is linearly separable
        """
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        # Initialize random number generator
        rgen = np.random.RandomState(self.random_state)
        
        # Initialize weights: [bias, w1, w2, ..., wn]
        # Small random values help break symmetry
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # Track errors per epoch
        self.errors_ = []
        
        # Training loop
        for epoch in range(self.n_iterations):
            errors = 0
            
            # Iterate through all training examples
            for xi, yi in zip(X, y):
                # Compute prediction
                prediction = self.predict(xi.reshape(1, -1))[0]
                
                # Compute error
                error = yi - prediction
                
                # Update weights if misclassified
                if error != 0:
                    # Update rule: w ← w + α * error * x
                    # weights_[0] is bias, weights_[1:] are feature weights
                    self.weights_[0] += self.learning_rate * error
                    self.weights_[1:] += self.learning_rate * error * xi
                    errors += 1
            
            # Record errors for this epoch
            self.errors_.append(errors)
            
            # Early stopping if converged (no errors)
            if errors == 0:
                break
        
        return self
    
    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the net input (weighted sum + bias).
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Net input z = w^T x + b of shape (n_samples,)
            
        Mathematical Formulation:
            z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
              = w^T x + b
        """
        # weights_[0] is bias, weights_[1:] are feature weights
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,), values in {0, 1}
            
        Raises:
            ValueError: If model hasn't been trained yet
            ValueError: If X has wrong number of features
            
        Mathematical Formulation:
            ŷ = σ(z) where σ(z) = 1 if z ≥ 0, else 0
        """
        if self.weights_ is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before predict()."
            )
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Validate feature count
        expected_features = len(self.weights_) - 1
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"but got {X.shape[1]}"
            )
        
        # Apply step function: return 1 if net_input >= 0, else 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy on test data.
        
        Args:
            X: Test features of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            
        Returns:
            Accuracy score between 0 and 1
            
        Example:
            >>> accuracy = model.score(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.2%}")
            Accuracy: 95.00%
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
            Dictionary containing weights, bias, and training history
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.weights_ is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before get_params()."
            )
        
        return {
            'bias': self.weights_[0],
            'weights': self.weights_[1:],
            'errors_per_epoch': self.errors_,
            'converged': self.errors_[-1] == 0 if self.errors_ else False,
            'n_epochs_trained': len(self.errors_)
        }
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (net input without thresholding).
        
        Useful for visualizing decision boundaries and confidence.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Decision function values of shape (n_samples,)
            Positive values indicate class 1, negative values indicate class 0
            
        Notes:
            Unlike predict(), this returns continuous values.
            The decision boundary is where decision_function(X) = 0.
        """
        if self.weights_ is None:
            raise ValueError(
                "Model has not been trained yet. Call fit() before decision_function()."
            )
        
        return self.net_input(X)
    
    def __repr__(self) -> str:
        """String representation of the perceptron."""
        return (
            f"Perceptron(learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, "
            f"random_state={self.random_state})"
        )
