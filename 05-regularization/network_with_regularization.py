# Extended network class with dropout and weight decay support

import numpy as np
from regularization import Dropout, compute_weight_decay_penalty, apply_weight_decay_to_gradients


class RegularizedNetwork:
    """Neural network with dropout and weight decay support.
    
    Extends the MNIST network from 04-optimization/ with regularization capabilities.
    Architecture: 784 (input) → hidden_size (default 256) → 10 (output)
    Uses ReLU activation for hidden layer and softmax for output layer.
    Supports dropout and weight decay for regularization.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 output_size: int = 10, dropout_rate: float = 0.0, seed: int = None):
        """Initialize regularized network with Xavier weight initialization.
        
        Args:
            input_size: Number of input neurons (default 784 for MNIST)
            hidden_size: Number of hidden layer neurons (default 256)
            output_size: Number of output neurons (default 10 for digits 0-9)
            dropout_rate: Dropout rate for regularization (0.0 to 1.0, default 0.0)
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.dropout_rate = dropout_rate
        self.training_mode = True
        
        # Xavier initialization: scale by sqrt(2 / n_in) for better gradient flow
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros(output_size)
        }
        
        # Initialize dropout layer
        self.dropout = Dropout(dropout_rate)
        
        self.cache = {}  # Store intermediate values for backpropagation
    
    def set_training_mode(self, training: bool):
        """Set whether network is in training or inference mode.
        
        During training: dropout is active
        During inference: dropout is inactive (all neurons active)
        
        Args:
            training: True for training mode, False for inference mode
        """
        self.training_mode = training
    
    def relu(self, x):
        """ReLU activation function: max(0, x).
        
        Args:
            x: Input array
            
        Returns:
            Array with ReLU applied element-wise
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU: 1 if x > 0, else 0.
        
        Args:
            x: Input array (pre-activation values)
            
        Returns:
            Array with derivative values
        """
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Numerically stable softmax activation.
        
        Subtracts max value before exponential to prevent overflow.
        
        Args:
            x: Input array of shape (batch_size, num_classes)
            
        Returns:
            Array of same shape with softmax applied, values sum to 1 per row
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation through the network with dropout.
        
        Computes: X → (W1, b1) → ReLU → Dropout → (W2, b2) → Softmax
        Dropout is only applied during training mode.
        Caches intermediate values for backpropagation.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, output_size)
        """
        # Layer 1: Linear transformation + ReLU
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.relu(z1)
        
        # Apply dropout after ReLU activation
        a1_dropped = self.dropout.forward(a1, training=self.training_mode)
        
        # Layer 2: Linear transformation + Softmax
        z2 = a1_dropped @ self.params['W2'] + self.params['b2']
        a2 = self.softmax(z2)
        
        # Cache values for backpropagation
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'a1_dropped': a1_dropped,
            'z2': z2,
            'a2': a2
        }
        
        return a2
    
    def backward(self, y_true):
        """Backward propagation to compute gradients, accounting for dropout.
        
        Uses cached values from forward pass to compute gradients
        for all parameters (W1, b1, W2, b2).
        Backpropagates through the dropout layer.
        
        Args:
            y_true: True labels, one-hot encoded, shape (batch_size, output_size)
            
        Returns:
            Dictionary of gradients for all parameters
        """
        batch_size = y_true.shape[0]
        
        # Output layer gradient (softmax + cross-entropy derivative)
        # For softmax + cross-entropy, derivative simplifies to: y_pred - y_true
        dz2 = self.cache['a2'] - y_true
        dW2 = self.cache['a1_dropped'].T @ dz2 / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # Hidden layer gradient (backpropagate through dropout)
        da1_dropped = dz2 @ self.params['W2'].T
        da1 = self.dropout.backward(da1_dropped)
        dz1 = da1 * self.relu_derivative(self.cache['z1'])
        dW1 = self.cache['X'].T @ dz1 / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size
        
        gradients = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
        
        return gradients
    
    def compute_loss(self, y_pred, y_true, weight_decay: float = 0.0):
        """Compute cross-entropy loss with optional weight decay penalty.
        
        Total loss = cross-entropy loss + weight decay penalty
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, output_size)
            y_true: True labels, one-hot encoded, shape (batch_size, output_size)
            weight_decay: Weight decay coefficient (default 0.0, no regularization)
            
        Returns:
            Tuple of (total_loss, ce_loss, penalty)
                - total_loss: Cross-entropy loss + weight decay penalty
                - ce_loss: Cross-entropy loss only
                - penalty: Weight decay penalty only
        """
        batch_size = y_true.shape[0]
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # Cross-entropy: -mean(sum(y_true * log(y_pred)))
        ce_loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
        
        # Weight decay penalty
        penalty = 0.0
        if weight_decay > 0:
            penalty = compute_weight_decay_penalty(self.params, weight_decay)
        
        total_loss = ce_loss + penalty
        return total_loss, ce_loss, penalty
    
    def predict(self, X):
        """Make predictions on input data (inference mode, no dropout).
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Predicted class labels (integers 0-9) of shape (batch_size,)
        """
        # Temporarily switch to inference mode
        original_mode = self.training_mode
        self.set_training_mode(False)
        y_pred = self.forward(X)
        self.set_training_mode(original_mode)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y_true_labels):
        """Compute classification accuracy.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            y_true_labels: True class labels (integers), shape (batch_size,)
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        predictions = self.predict(X)
        return np.mean(predictions == y_true_labels)


class RegularizedTrainer:
    """Trainer class for training regularized networks with weight decay support.
    
    Handles the complete training loop including mini-batch processing,
    weight updates via optimizer, dropout, weight decay, and metrics tracking.
    """
    
    def __init__(self, network: RegularizedNetwork, optimizer, weight_decay: float = 0.0):
        """Initialize trainer with network, optimizer, and weight decay.
        
        Args:
            network: RegularizedNetwork instance to train
            optimizer: Optimizer instance (VanillaGD, SGD, MomentumSGD, or Adam)
            weight_decay: Weight decay coefficient (default 0.0, no regularization)
        """
        self.network = network
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'ce_loss': [],      # Cross-entropy loss without penalty
            'penalty': []       # Weight decay penalty
        }
    
    def train(self, X_train, y_train_onehot, y_train_labels,
              X_test, y_test_onehot, y_test_labels,
              epochs: int, batch_size: int, verbose: bool = True):
        """Train the network using mini-batch gradient descent with regularization.
        
        Args:
            X_train: Training data, shape (n_samples, input_size)
            y_train_onehot: Training labels (one-hot), shape (n_samples, output_size)
            y_train_labels: Training labels (integers), shape (n_samples,)
            X_test: Test data, shape (n_test, input_size)
            y_test_onehot: Test labels (one-hot), shape (n_test, output_size)
            y_test_labels: Test labels (integers), shape (n_test,)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print progress (default True)
            
        Returns:
            Dictionary containing training history with keys:
                'train_loss', 'train_acc', 'test_loss', 'test_acc', 'ce_loss', 'penalty'
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data at start of each epoch
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_penalty = 0
            
            # Mini-batch training loop
            for batch_idx in range(n_batches):
                # Extract mini-batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass: compute predictions and loss
                self.network.set_training_mode(True)
                y_pred = self.network.forward(X_batch)
                total_loss, ce_loss, penalty = self.network.compute_loss(y_pred, y_batch, self.weight_decay)
                
                epoch_loss += total_loss
                epoch_ce_loss += ce_loss
                epoch_penalty += penalty
                
                # Backward pass: compute gradients
                gradients = self.network.backward(y_batch)
                
                # Apply weight decay to gradients
                if self.weight_decay > 0:
                    gradients = apply_weight_decay_to_gradients(
                        gradients, self.network.params, self.weight_decay
                    )
                
                # Update weights using optimizer
                self.network.params = self.optimizer.update(self.network.params, gradients)
            
            # Compute metrics after each epoch
            avg_train_loss = epoch_loss / n_batches
            avg_ce_loss = epoch_ce_loss / n_batches
            avg_penalty = epoch_penalty / n_batches
            
            train_acc = self.network.accuracy(X_train, y_train_labels)
            
            # Compute test metrics
            self.network.set_training_mode(False)
            y_test_pred = self.network.forward(X_test)
            test_total_loss, test_ce_loss, test_penalty = self.network.compute_loss(y_test_pred, y_test_onehot, self.weight_decay)
            test_acc = self.network.accuracy(X_test, y_test_labels)
            
            # Store metrics in history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_total_loss)
            self.history['test_acc'].append(test_acc)
            self.history['ce_loss'].append(avg_ce_loss)
            self.history['penalty'].append(avg_penalty)
            
            # Optional verbose output
            if verbose:
                gap = train_acc - test_acc
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - "
                      f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f} - Gap: {gap:.4f}")
        
        return self.history
