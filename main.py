import backPropgation
from activationFunctions import relu, sigmoid, tanH, linear, softMax
from lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction
from optimisers import Optimisers
from initialisers import Initialisers

#weights are in [number of layers][size of current layer][size of next layer]
#biases are in [number of layers][current node in the layer]

class Network(Optimisers, Initialisers):
    def __init__(self, lossFunc = "MSE", learningRate = 0.01, optimisationFunc = "gd", useMomentum = False, momentumCoefficient = 0.9, momentumDecay = 0.99, useBatches = False, batchSize = 32):
        self.layers = []
        self.weights = []
        self.biases = []
        self.learningRate = learningRate

        lossFunctionDict = {
            "mse": MSELossFunction,
            "mae": MAELossFunction,
            "cross entropy": CrossEntropyLossFunction,
        }
        self.lossFunction = lossFunctionDict[lossFunc.lower()]

        optimisationFunctionDict = {
            "gd": self.trainGradientDescent,
            "sgd": self.trainStochasticGradientDescent,
            "batching": self.trainGradientDescentUsingBatching, "batches": self.trainGradientDescentUsingBatching, 
        }
        self.optimisationFunction = optimisationFunctionDict[optimisationFunc.lower()]
        if(useBatches == True):
            self.optimisationFunction = self.trainGradientDescentUsingBatching

        self.useMomentum = useMomentum
        self.momentumCoefficient = momentumCoefficient
        self.momentumDecay = momentumDecay
        self.velocityWeight = None
        self.velocityBias = None

        self.useBatches = useBatches
        self.batchSize = batchSize

    def addLayer(self, size, activationFunction):
        funcs = {
            "relu": relu,
            "sigmoid": sigmoid,
            "linear": linear,
            "tanh": tanH,
            "softmax": softMax,
        }
        if(activationFunction.lower() not in funcs):
            raise ValueError(f"Activation function not made: {activationFunction.lower()}")
        self.layers.append([size, funcs[activationFunction.lower()]])

    def calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currentLayer):
        layer = []
        for i in range(currentLayer[0]):
            summ = biases[i]
            for j in range(len(lastLayerNodes)):
                summ += lastLayerNodes[j] * lastLayerWeights[i][j]
            if(currentLayer[1] != self.softMax):
                layer.append(currentLayer[1](summ))
            else:
                layer.append(summ)
        if(currentLayer[1] == self.softMax):
            return self.softMax(layer)
        return layer
    
    def forwardPropagation(self, inputData):
        layerNodes = [inputData]
        for i in range(1, len(self.layers)):
            layerNodes.append(self.calculateLayerNodes(layerNodes[i - 1], self.weights[i - 1], self.biases[i - 1], self.layers[i]))
        return layerNodes
    
    def backPropgation(self, layerNodes, weights, biases, trueValues):
        weightGradients, biasGradients = backPropgation.backPropgation(layerNodes, weights, biases, trueValues, self.layers, self.lossFunction, self.learningRate)
        return weightGradients, biasGradients

    def train(self, inputData, labels, epochs):
        self.optimisationFunction(inputData, labels, epochs)
