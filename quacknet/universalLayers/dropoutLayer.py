import numpy as np

class Dropout:
    def __init__(self, dropProbability=0.5):
        self.dropProbability = dropProbability
        self.mask = None

    def forward(self, inputTensor, training = True):
        if(training == False):
            return inputTensor
        
        if(self.dropProbability == 0):
            self.mask = np.ones_like(inputTensor)
            return inputTensor
        elif(self.dropProbability == 1):
            self.mask = np.zeros_like(inputTensor)
            return np.zeros_like(inputTensor)

        self.mask = (np.random.rand(*inputTensor.shape) > self.dropProbability).astype(inputTensor.dtype)
        return inputTensor * self.mask / (1 - self.dropProbability)

    def _backpropagation(self, gradient):
        return gradient * self.mask / (1 - self.dropProbability)