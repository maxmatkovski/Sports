# example neural network from scratch
import numpy as np

#  create NeuralNetwork Class
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # initialize weights and biases
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros(1,self.output_dim)

    def forward(self, X):
        # compute the hidden layer activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # compute the output layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # Number of samples
        
        # Compute gradients
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            outputs = self.forward(X)
            
            # Backward propagation
            self.backward(X, y, learning_rate)
            
            # Compute loss (optional)
            loss = self.calculate_loss(y, outputs)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")
    
    def predict(self, X):
        # Forward propagation
        outputs = self.forward(X)
        predictions = np.round(outputs)
        
        return predictions
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def calculate_loss(self, y_true, y_pred):
        epsilon = 1e-7  # Small value to avoid division by zero
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        
        return loss