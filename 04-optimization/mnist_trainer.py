# MNIST training infrastructure for blog post #4

import numpy as np


def load_mnist():
    """Load MNIST dataset using keras datasets (for convenience).
    
    Returns:
        tuple: ((X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot))
            - X_train: Training images, shape (60000, 784), normalized to [0, 1]
            - y_train: Training labels, shape (60000,), integer values 0-9
            - y_train_onehot: Training labels one-hot encoded, shape (60000, 10)
            - X_test: Test images, shape (10000, 784), normalized to [0, 1]
            - y_test: Test labels, shape (10000,), integer values 0-9
            - y_test_onehot: Test labels one-hot encoded, shape (10000, 10)
    """
    try:
        # Try tensorflow/keras first
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except ImportError:
        try:
            # Try standalone keras
            from keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        except ImportError:
            # Fallback to scikit-learn's fetch_openml
            from sklearn.datasets import fetch_openml
            print("Loading MNIST from scikit-learn (this may take a moment on first run)...")
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X = mnist.data.astype('float32')
            y = mnist.target.astype('int64')
            
            # Split into train/test (first 60000 for train, rest for test)
            X_train = X[:60000]
            y_train = y[:60000]
            X_test = X[60000:]
            y_test = y[60000:]
            
            # Data is already flattened from fetch_openml, but needs to be numpy arrays
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
    
    # Ensure data is in the right format
    if len(X_train.shape) == 3:
        # Flatten images: 28x28 -> 784
        X_train = X_train.reshape(-1, 784).astype('float32')
        X_test = X_test.reshape(-1, 784).astype('float32')
    
    # Normalize pixel values to [0, 1]
    if X_train.max() > 1.0:
        X_train /= 255.0
        X_test /= 255.0
    
    # One-hot encode labels
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    
    return (X_train, y_train, y_train_onehot), (X_test, y_test, y_test_onehot)


class Network:
    """Neural network for MNIST classification.
    
    Architecture: 784 (input) → hidden_size (default 128) → 10 (output)
    Uses ReLU activation for hidden layer and softmax for output layer.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10, seed: int = None):
        """Initialize network with Xavier weight initialization.
        
        Args:
            input_size: Number of input neurons (default 784 for MNIST)
            hidden_size: Number of hidden layer neurons (default 128)
            output_size: Number of output neurons (default 10 for digits 0-9)
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Xavier initialization: scale by sqrt(2 / n_in) for better gradient flow
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros(output_size)
        }
        
        self.cache = {}  # Store intermediate values for backpropagation
    
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
        """Forward propagation through the network.
        
        Computes: X → (W1, b1) → ReLU → (W2, b2) → Softmax
        Caches intermediate values for backpropagation.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, output_size)
        """
        # Layer 1: Linear transformation + ReLU
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.relu(z1)
        
        # Layer 2: Linear transformation + Softmax
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = self.softmax(z2)
        
        # Cache values for backpropagation
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2
    
    def backward(self, y_true):
        """Backward propagation to compute gradients.
        
        Uses cached values from forward pass to compute gradients
        for all parameters (W1, b1, W2, b2).
        
        Args:
            y_true: True labels, one-hot encoded, shape (batch_size, output_size)
            
        Returns:
            Dictionary of gradients for all parameters
        """
        batch_size = y_true.shape[0]
        
        # Output layer gradient (softmax + cross-entropy derivative)
        # For softmax + cross-entropy, derivative simplifies to: y_pred - y_true
        dz2 = self.cache['a2'] - y_true
        dW2 = self.cache['a1'].T @ dz2 / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size
        
        # Hidden layer gradient
        da1 = dz2 @ self.params['W2'].T
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
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss with numerical stability.
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, output_size)
            y_true: True labels, one-hot encoded, shape (batch_size, output_size)
            
        Returns:
            Scalar loss value (average over batch)
        """
        batch_size = y_true.shape[0]
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # Cross-entropy: -mean(sum(y_true * log(y_pred)))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
        
        return loss
    
    def predict(self, X):
        """Make predictions on input data.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Predicted class labels (integers 0-9) of shape (batch_size,)
        """
        y_pred = self.forward(X)
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



class Trainer:
    """Trainer class for training neural networks with mini-batch support.
    
    Handles the complete training loop including mini-batch processing,
    weight updates via optimizer, and metrics tracking.
    """
    
    def __init__(self, network: Network, optimizer):
        """Initialize trainer with network and optimizer.
        
        Args:
            network: Network instance to train
            optimizer: Optimizer instance (VanillaGD, SGD, MomentumSGD, or Adam)
        """
        self.network = network
        self.optimizer = optimizer
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

    
    def train(self, X_train, y_train_onehot, y_train_labels,
              X_test, y_test_onehot, y_test_labels,
              epochs: int, batch_size: int, verbose: bool = True):
        """Train the network using mini-batch gradient descent.
        
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
                'train_loss', 'train_acc', 'test_loss', 'test_acc'
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data at start of each epoch
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            
            # Mini-batch training loop
            for batch_idx in range(n_batches):
                # Extract mini-batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass: compute predictions and loss
                y_pred = self.network.forward(X_batch)
                loss = self.network.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass: compute gradients
                gradients = self.network.backward(y_batch)
                
                # Update weights using optimizer
                self.network.params = self.optimizer.update(self.network.params, gradients)
            
            # Compute metrics after each epoch
            avg_train_loss = epoch_loss / n_batches
            train_acc = self.network.accuracy(X_train, y_train_labels)
            
            # Compute test metrics
            y_test_pred = self.network.forward(X_test)
            test_loss = self.network.compute_loss(y_test_pred, y_test_onehot)
            test_acc = self.network.accuracy(X_test, y_test_labels)
            
            # Store metrics in history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            # Optional verbose output
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_train_loss:.4f} - "
                      f"Train Acc: {train_acc:.4f} - "
                      f"Test Acc: {test_acc:.4f}")
        
        return self.history
