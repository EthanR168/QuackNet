from . import backPropgation
from .activationFunctions import relu, sigmoid, tanH, linear, softMax
from .lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction
from .optimisers import Optimisers
from .initialisers import Initialisers
from .writeAndReadWeightBias import writeAndRead
from .convulationalManager import CNNModel
import numpy as np

class Network(Optimisers, Initialisers, writeAndRead, CNNModel):
    def __init__(self, lossFunc = "MSE", learningRate = 0.01, optimisationFunc = "gd", useMomentum = False, momentumCoefficient = 0.9, momentumDecay = 0.99, useBatches = False, batchSize = 32):
        self.layers = []
        self.weights = []
        self.biases = []
        self.learningRate = learningRate

        lossFunctionDict = {
            "mse": MSELossFunction,
            "mae": MAELossFunction,
            "cross entropy": CrossEntropyLossFunction,"cross": CrossEntropyLossFunction,
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
    
    def backPropgation(self, layerNodes, weights, biases, trueValues, returnErrorTermForCNN = False):
        return backPropgation.backPropgation(layerNodes, weights, biases, trueValues, self.layers, self.lossFunction, returnErrorTermForCNN)
    
    def train(self, inputData, labels, epochs):
        self.checkIfNetworkCorrect()
        correct = 0
        totalLoss = 0
        nodes, self.weights, self.biases, self.velocityWeight, self.velocityBias = self.optimisationFunction(inputData, labels, epochs, self.weights, self.biases, self.momentumCoefficient, self.momentumDecay, self.useMomentum, self.velocityWeight, self.velocityBias, self.learningRate, self.batchSize)        
        lastLayer = len(nodes[0]) - 1
        for i in range(len(nodes)): 
            totalLoss += self.lossFunction(nodes[i][lastLayer], labels[i])
            nodeIndex = np.argmax(nodes[i][lastLayer])
            labelIndex = np.argmax(labels[i])
            if(nodeIndex == labelIndex):
                correct += 1
        return correct / (len(labels) * epochs), totalLoss / (len(labels) * epochs)
    
    def checkIfNetworkCorrect(self): #this is to check if activation functions/loss functions adhere to certain rule
        for i in range(len(self.layers) - 1): #checks if softmax is used for any activation func that isnt output layer
            if(self.layers[i][1] == softMax): #if so it stops the user
                raise ValueError(f"Softmax shouldnt be used in non ouput layers. Error at Layer {i + 1}")
        usingSoftMax = self.layers[len(self.layers) - 1][1] == softMax
        if(usingSoftMax == True):
            if(self.lossFunction != CrossEntropyLossFunction): #checks if softmax is used without cross entropy loss function
                raise ValueError(f"Softmax output layer requires Cross Entropy loss function") #if so stops the user
        elif(self.lossFunction == CrossEntropyLossFunction):
            raise ValueError(f"Cross Entropy loss function requires Softmax output layer") #if so stops the user
