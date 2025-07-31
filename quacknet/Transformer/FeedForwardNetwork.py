import numpy as np

"""
This Feed Forward Network (FFN) is a NN which is applied to each token independantly
But it is the same FFN each time (so it has the same weight and bias)

Number of Layers are hardcoded to be 2, to make code easier to follow
Also ReLU is used as the activation function (may allow users to set the activation function, in the future)
"""

class FeedForwardNetwork:
    def __init__(self, inputDimension, hiddenDimension, W1, b1, W2, b2):
        self.inputDimension = inputDimension
        self.hiddenDimension = hiddenDimension
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

        self.createWeights()

    def createWeights(self):
        self.W1 = self._initiaseWeight(self.inputDimension, self.hiddenDimension)
        self.W2 = self._initiaseWeight(self.hiddenDimension, self.inputDimension)
        
        self.b1 = np.zeros((1, self.hiddenDimension))
        self.b2 = np.zeros((1, self.inputDimension))

    def _initiaseWeight(self, inputDimension, outputDimension):
        return np.random.rand(inputDimension, outputDimension) * (1 / np.sqrt(inputDimension))
    

    def forwardPropagation(self, inputTokens): 
        firstLayer = np.matmul(inputTokens, self.W1) + self.b1
        activated = np.maximum(0, firstLayer) # ReLU
        output = np.matmul(activated, self.W2) + self.b2
        return output