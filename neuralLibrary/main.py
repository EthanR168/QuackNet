from . import backPropgation
from .activationFunctions import relu, sigmoid, tanH, linear, softMax
from .lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction
from .optimisers import Optimisers
from .initialisers import Initialisers
import numpy as np

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

    def addLayer(self, size, activationFunction="relu"):
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

    def calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currentLayer) -> np.ndarray:
        summ = np.dot(lastLayerNodes, lastLayerWeights) + biases
        if(currentLayer[1] != softMax):
            return currentLayer[1](summ)
        else:
            return softMax(summ)
        
    def forwardPropagation(self, inputData) -> list[np.ndarray]:
        layerNodes = [np.array(inputData)]
        for i in range(1, len(self.layers)):
            layerNodes.append(np.array(self.calculateLayerNodes(layerNodes[i - 1], self.weights[i - 1], self.biases[i - 1], self.layers[i])))
        return layerNodes
    
    def backPropgation(self, layerNodes, weights, biases, trueValues):
        weightGradients, biasGradients = backPropgation.backPropgation(layerNodes, weights, biases, trueValues, self.layers, self.lossFunction)
        return weightGradients, biasGradients

    def train(self, inputData, labels, epochs):
        self.optimisationFunction(inputData, labels, epochs, self.weights, self.biases, self.momentumCoefficient, self.momentumDecay, self.useMomentum, self.velocityWeight, self.velocityBias, self.learningRate, self.batchSize)

'''        
n = Network()
n.addLayer(3)
n.addLayer(2)
n.addLayer(1)
n.createWeightsAndBiases()
n.train([[0.5, 0.5, 0.5]], np.array([1]), 1)
'''