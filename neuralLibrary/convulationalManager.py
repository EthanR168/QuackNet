from .convulationalFeutures import ConvulationalNetwork
from .convulationalBackpropagation import CNNbackpropagation

class CNNModel():
    def __init__(self):
        self.layers = []
    
    def addLayer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(self, inputTensor)
        return inputTensor

    def backpropagation(self, allTensors):
        errorTerms = self.layers[0].backpropagation() # <-- this is a neural network
        for i in range(len(self.layers) - 2, -1, -1):
            if(self.layers[i] == ConvLayer or self.layers[i] == PoolingLayer):
                self.layers[i].backpropagation(errorTerms, allTensors[len(allTensors) - i])

class ConvLayer(CNNbackpropagation):
    def __init__(self, kernalSize, kernals, kernalBiases, numKernals, stride, padding = "no"):
        self.kernalSize = kernalSize
        self.numKernals = numKernals
        self.kernalSize = kernals
        self.numKernals = kernalBiases
        self.stride = stride
        self.padding = padding
        if(padding.lower() == "no"):
            self.usePadding = False
        else:
            self.padding = int(self.padding)
            self.usePadding = True
    
    def forward(self, inputTensor):
        return ConvulationalNetwork.kernalisation(self, inputTensor, self.kernals, self.kernalBiases, self.kernalSize, self.usePadding, self.padding, self.stride)

    def backpropagation(self, errorPatch, inputTensor):
        return CNNbackpropagation.ConvolutionDerivative(errorPatch, self.kernals, inputTensor, self.stride)

class PoolingLayer(CNNbackpropagation):
    def __init__(self, size, stride, mode = "max"):
        self.size = size
        self.stride = stride
        self.mode = mode
    
    def forward(self, inputTensor):
        return ConvulationalNetwork.pooling(self, inputTensor, self.size, self.stride, self.mode)

    def backpropagation(self, errorPatch, inputTensor):
        if(self.mode == "max"):
            return CNNbackpropagation.MaxPoolingDerivative(self, errorPatch, inputTensor, self.size, self.stride)
        else:
            return CNNbackpropagation.AveragePoolingDerivative(self, errorPatch, inputTensor, self.size, self.stride)

class DenseLayer: # basically a fancy neural network
    def __init__(self, NeuralNetworkClass):
        self.NeuralNetworkClass = NeuralNetworkClass
    
    def forward(self, inputTensor):
        inputArray = ConvulationalNetwork.flatternTensor(self, inputTensor)
        self.NeuralNetworkClass.forward(self, inputArray)
    
    def backpropagation(self):
        raise ValueError("Couldn not be bothered")
