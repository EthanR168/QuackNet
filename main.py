import math, random
import backPropgation

#weights are in [number of layers][size of current layer][size of next layer]
#biases are in [number of layers][current node in the layer]

class Network:
    def __init__(self, lossFunc = "MSE", learningRate = 0.01, optimisationFunc = "gd"):
        self.layers = []
        self.weights = []
        self.biases = []
        self.learningRate = learningRate

        lossFunctionDict = {
            "mse": self.MSELossFunction,
            "mae": self.MAELossFunction,
        }
        self.lossFunction = lossFunctionDict[lossFunc.lower()]

        optimisationFunctionDict = {
            "gd": self.trainGradientDescent,
            "sgd": self.trainStochasticGradientDescent,
        }
        self.optimisationFunction = optimisationFunctionDict[optimisationFunc.lower()]

    def createWeightsAndBiases(self):
        #weights are in [number of layers][size of current layer][size of next layer]
        for i in range(1, len(self.layers)):
            currSize = self.layers[i][0]
            lastSize = self.layers[i - 1][0]
            actFunc = self.layers[i][1]

            if(actFunc == self.relu):
                bounds =  math.sqrt(2 / lastSize) # He initialisation
            elif(actFunc == self.sigmoid):
                bounds = math.sqrt(6/ (lastSize + currSize)) # Xavier initialisation
            else:
                bounds = 1
                
            currW = []
            for _ in range(currSize):
                w = []
                for _ in range(lastSize):
                    w.append(random.gauss(0, bounds)) 
                currW.append(w)
            self.weights.append(currW)

            b = []
            for _ in range(currSize):
                b.append(0)
            self.biases.append(b)

    def MSELossFunction(self, predicted, true):
        summ = 0
        for i in range(len(predicted)):
            summ += (true[i] - predicted[i]) ** 2
        return summ / len(predicted)

    def MAELossFunction(self, predicted, true):
        summ = 0
        for i in range(len(predicted)):
            summ += abs(true[i] - predicted[i])
        return summ / len(predicted)

    def relu(self, value):
        return max(0, value)
    
    def sigmoid(self, value):
        return (1 / (1 + math.exp(-value)))
    
    def linear(self, value): # <-- Warning very computationaly expsensive, so be wary when using
        return value

    def addLayer(self, size, activationFunction):
        funcs = {
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "linear": self.linear,
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
            layer.append(currentLayer[1](summ))
        return layer
    
    def forwardPropagation(self, inputData):
        layerNodes = [inputData]
        for i in range(1, len(self.layers)):
            layerNodes.append(self.calculateLayerNodes(layerNodes[i - 1], self.weights[i - 1], self.biases[i - 1], self.layers[i]))
        return layerNodes
    
    def backPropgation(self, layerNodes, weights, biases, trueValues):
        weightGradients, biaseGradients = backPropgation.backPropgation(layerNodes, weights, biases, trueValues, self.layers, self.lossFunction, self.learningRate)
        return weightGradients, biaseGradients

    def train(self, inputData, labels, epochs):
        self.optimisationFunction(inputData, labels, epochs)

    def trainGradientDescent(self, inputData, labels, epochs):
        for _ in range(epochs):
            weightGradients, biaseGradients = [], []
            for i in range(len(self.weights)):
                ww = []
                for j in range(len(self.weights[i])):
                    w = []
                    for a in range(len(self.weights[i][j])):
                        w.append(0)
                    ww.append(w)
                weightGradients.append(ww)                    
            for i in range(len(self.biases)):
                b = []
                for j in range(len(self.biases[i])):
                    b.append(0)
                biaseGradients.append(b)

            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)

                for i in range(len(w)):
                    for j in range(len(w[i])):
                        for a in range(len(w[i][j])):
                            weightGradients[i][j][a] += w[i][j][a]
                            
                for i in range(len(b)):
                    for j in range(len(b[i])):
                        biaseGradients[i][j] += b[i][j]

            for i in range(len(weightGradients)):
                for j in range(len(weightGradients[i])):
                    for a in range(len(weightGradients[i][j])):
                        self.weights[i][j][a] -= self.learningRate * (weightGradients[i][j][a] / len(inputData))
                            
            for i in range(len(biaseGradients)):
                for j in range(len(biaseGradients[i])):
                    self.biases[i][j] -= self.learningRate * (biaseGradients[i][j] / len(inputData))

    def trainStochasticGradientDescent(self, inputData, labels, epochs):
        for _ in range(epochs):
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)

                for i in range(len(self.weights)):
                    for j in range(len(self.weights[i])):
                        for a in range(len(self.weights[i][j])):
                            self.weights[i][j][a] -= self.learningRate * w[i][j][a]
                                
                for i in range(len(self.biases)):
                    for j in range(len(self.biases[i])):
                        self.biases[i][j] -= self.learningRate * b[i][j]
                