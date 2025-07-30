import numpy as np

class NormLayer:
    def __init__(self, features, epsilon = 1e-6):
        self.gamma = np.ones((1, 1, features)) # scale
        self.beta = np.zeros((1, 1, features)) # shift
        self.epsilon = epsilon # adds a small constant to avoid division by 0
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True) 
        variance = np.var(x, axis=-1, keepdims=True)
        normalised = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * normalised + self.beta