from .convulationalFeutures import ConvulationalNetwork
from .convulationalBackpropagation import CNNbackpropagation
from .activationDerivativeFunctions import ReLUDerivative
import numpy as np

class CNNModel():
    def __init__(self):
        self.layers = []
    
    def addLayer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(inputTensor)
        return inputTensor

    def backpropagation(self, allTensors):
        errorTerms = self.layers[0].backpropagation() # <-- this is a neural network
        for i in range(len(self.layers) - 2, -1, -1):
            if(self.layers[i] == ConvLayer or self.layers[i] == PoolingLayer):
                self.layers[i].backpropagation(errorTerms, allTensors[len(allTensors) - i])

class ConvLayer(ConvulationalNetwork, CNNbackpropagation):
    def __init__(self, kernalSize, kernalWeights, kernalBiases, numKernals, stride, padding = "no"):
        self.kernalSize = kernalSize
        self.numKernals = numKernals
        self.kernalWeights = kernalWeights
        self.kernalBiases = kernalBiases
        self.stride = stride
        self.padding = padding
        if(padding.lower() == "no"):
            self.usePadding = False
        else:
            self.padding = int(self.padding)
            self.usePadding = True
    
    def forward(self, inputTensor):
        return ConvulationalNetwork.kernalisation(self, inputTensor, self.kernalWeights, self.kernalBiases, self.kernalSize, self.usePadding, self.padding, self.stride)

    def backpropagation(self, errorPatch, inputTensor):
        return CNNbackpropagation.ConvolutionDerivative(errorPatch, self.kernals, inputTensor, self.stride)

class PoolingLayer(CNNbackpropagation):
    def __init__(self, gridSize, stride, mode = "max"):
        self.gridSize = gridSize
        self.stride = stride
        self.mode = mode
    
    def forward(self, inputTensor):
        return ConvulationalNetwork.pooling(self, inputTensor, self.gridSize, self.stride, self.mode)

    def backpropagation(self, errorPatch, inputTensor):
        if(self.mode == "max"):
            return CNNbackpropagation.MaxPoolingDerivative(self, errorPatch, inputTensor, self.gridSize, self.stride)
        else:
            return CNNbackpropagation.AveragePoolingDerivative(self, errorPatch, inputTensor, self.gridSize, self.stride)

class DenseLayer: # basically a fancy neural network
    def __init__(self, NeuralNetworkClass):
        self.NeuralNetworkClass = NeuralNetworkClass
    
    def forward(self, inputTensor):
        inputArray = ConvulationalNetwork.flatternTensor(self, inputTensor)
        return np.array(self.NeuralNetworkClass.forwardPropagation(inputArray)[-1])
    
    def backpropagation(self):
        raise ValueError("Couldnt not be bothered, yet ...")

class ActivationLayer: # basically aplies an activation function over the whole network (eg. leaky relu)
    def forward(self, inputTensor):
        return ConvulationalNetwork.activation(self, inputTensor)

    def backpropagation(self, errorPatch, inputTensor):
        return CNNbackpropagation.ActivationLayerDerivative(self, errorPatch, ReLUDerivative, inputTensor)