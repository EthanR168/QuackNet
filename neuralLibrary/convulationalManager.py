from .convulationalFeutures import ConvulationalNetwork

class CNNModel:
    def __init__(self):
        self.layers = []
    
    def addLayer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(self, inputTensor)
        return inputTensor

class ConvLayer:
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

class PoolingLayer:
    def __init__(self, size, stride, mode = "max"):
        self.size = size
        self.stride = stride
        self.mode = mode
    
    def forward(self, inputTensor):
        return ConvulationalNetwork.pooling(self, inputTensor, self.size, self.stride, self.mode)

class DenseLayer: # basically a fancy neural network
    def __init__(self, NeuralNetworkClass):
        self.NeuralNetworkClass = NeuralNetworkClass
    
    def forward(self, inputTensor):
        inputArray = ConvulationalNetwork.flatternTensor(self, inputTensor)
        self.NeuralNetworkClass.forward(self, inputArray)
