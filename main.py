import math, random
import backPropgation

#weights are in [number of layers][size of current layer][size of next layer]
#biases are in [number of layers][current node in the layer]

class Network:
    def __init__(self, lossFunc = "MSE", learningRate = 0.01, optimisationFunc = "gd", useMomentum = False, momentumCoefficient = 0.9, momentumDecay = 0.99, useBatches = False, batchSize = 32):
        self.layers = []
        self.weights = []
        self.biases = []
        self.learningRate = learningRate

        lossFunctionDict = {
            "mse": self.MSELossFunction,
            "mae": self.MAELossFunction,
            "cross entropy": self.CrossEntropyLossFunction,
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

    def CrossEntropyLossFunction(self, predicted, true):
        summ = 0
        for i in range(len(predicted)):
            summ += true[i] * math.log(predicted[i])
        return -summ

    def relu(self, value):
        return max(0, value)
    
    def sigmoid(self, value):
        return (1 / (1 + math.exp(-value)))
    
    def tanH(self, value):
        return ((math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value)))
    
    def linear(self, value): # <-- Warning very computationaly expsensive, so be wary when using
        return value

    def softMax(self, values): # gets the whole array of values
        summ, out = 0, []
        for i in values:
            summ += math.exp(i)
        for i in values:
            out.append(math.exp(i)/summ)
        return out

    def addLayer(self, size, activationFunction):
        funcs = {
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "linear": self.linear,
            "tanh": self.tanH,
            "softmax": self.softMax,
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

    def trainGradientDescent(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            if(self.velocityWeight == None):
                vel = []
                for i in range(len(self.weights)):
                    vv = []
                    for j in range(len(self.weights[i])):
                        v = []
                        for a in range(len(self.weights[i][j])):
                            v.append(0)
                        vv.append(v)
                    vel.append(vv) 
                self.velocityWeight = vel
            if(self.velocityBias == None):
                bb = []
                for i in range(len(self.biases)):
                    b = []
                    for j in range(len(self.biases[i])):
                        b.append(0)
                    bb.append(b)
                self.velocityBias = bb
        
        for _ in range(epochs):
            weightGradients, biasGradients = [], []
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
                biasGradients.append(b)

            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)

                for i in range(len(w)):
                    for j in range(len(w[i])):
                        for a in range(len(w[i][j])):
                            weightGradients[i][j][a] += w[i][j][a]
                            
                for i in range(len(b)):
                    for j in range(len(b[i])):
                        biasGradients[i][j] += b[i][j]

            for i in range(len(weightGradients)):
                for j in range(len(weightGradients[i])):
                    for a in range(len(weightGradients[i][j])):
                        if(self.useMomentum == True):
                            self.velocityWeight[i][j][a] = self.momentumCoefficient * self.velocityWeight[i][j][a] - self.learningRate * (weightGradients[i][j][a] / len(inputData))
                            self.weights[i][j][a] += self.velocityWeight[i][j][a]
                        else:
                            self.weights[i][j][a] -= self.learningRate * (weightGradients[i][j][a] / len(inputData))
                            
            for i in range(len(biasGradients)):
                for j in range(len(biasGradients[i])):
                    if(self.useMomentum == True):
                        self.velocityBias[i][j] = self.momentumCoefficient * self.velocityBias[i][j] - self.learningRate * (biasGradients[i][j] / len(inputData))
                        self.biases[i][j] += self.velocityBias[i][j]
                    else:
                        self.biases[i][j] -= self.learningRate * (biasGradients[i][j] / len(inputData))
            self.momentumCoefficient *= self.momentumDecay

    def trainStochasticGradientDescent(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            if(self.velocityWeight == None):
                vel = []
                for i in range(len(self.weights)):
                    vv = []
                    for j in range(len(self.weights[i])):
                        v = []
                        for a in range(len(self.weights[i][j])):
                            v.append(0)
                        vv.append(v)
                    vel.append(vv) 
                self.velocityWeight = vel
            if(self.velocityBias == None):
                bb = []
                for i in range(len(self.biases)):
                    b = []
                    for j in range(len(self.biases[i])):
                        b.append(0)
                    bb.append(b)
                self.velocityBias = bb
        
        for _ in range(epochs):
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)

                for i in range(len(self.weights)):
                    for j in range(len(self.weights[i])):
                        for a in range(len(self.weights[i][j])):
                            if(self.useMomentum == True):
                                self.velocityWeight[i][j][a] = self.momentumCoefficient * self.velocityWeight[i][j][a] - self.learningRate * w[i][j][a]
                                self.weights[i][j][a] += self.velocityWeight[i][j][a]
                            else:
                                self.weights[i][j][a] -= self.learningRate * w[i][j][a]
                                
                for i in range(len(self.biases)):
                    for j in range(len(self.biases[i])):
                        if(self.useMomentum == True):
                            self.velocityBias[i][j] = self.momentumCoefficient * self.velocityBias[i][j] - self.learningRate * b[i][j]
                            self.biases[i][j] += self.velocityBias[i][j]
                        else:
                            self.biases[i][j] -= self.learningRate * b[i][j]
                            
            self.momentumCoefficient *= self.momentumDecay

    def trainGradientDescentUsingBatching(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            if(self.velocityWeight == None):
                vel = []
                for i in range(len(self.weights)):
                    vv = []
                    for j in range(len(self.weights[i])):
                        v = []
                        for a in range(len(self.weights[i][j])):
                            v.append(0)
                        vv.append(v)
                    vel.append(vv) 
                self.velocityWeight = vel
            if(self.velocityBias == None):
                bb = []
                for i in range(len(self.biases)):
                    b = []
                    for j in range(len(self.biases[i])):
                        b.append(0)
                    bb.append(b)
                self.velocityBias = bb
        
        for _ in range(epochs):
            weightGradients, biasGradients = [], []
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
                biasGradients.append(b)

            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)

                for i in range(len(w)):
                    for j in range(len(w[i])):
                        for a in range(len(w[i][j])):
                            weightGradients[i][j][a] += w[i][j][a]
                            
                for i in range(len(b)):
                    for j in range(len(b[i])):
                        biasGradients[i][j] += b[i][j]

                if((data + 1) % self.batchSize == 0):
                    for i in range(len(weightGradients)):
                        for j in range(len(weightGradients[i])):
                            for a in range(len(weightGradients[i][j])):
                                if(self.useMomentum == True):
                                    self.velocityWeight[i][j][a] = self.momentumCoefficient * self.velocityWeight[i][j][a] - self.learningRate * (weightGradients[i][j][a] / self.batchSize)
                                    self.weights[i][j][a] += self.velocityWeight[i][j][a]
                                else:
                                    self.weights[i][j][a] -= self.learningRate * (weightGradients[i][j][a] / self.batchSize)
                                    
                    for i in range(len(biasGradients)):
                        for j in range(len(biasGradients[i])):
                            if(self.useMomentum == True):
                                self.velocityBias[i][j] = self.momentumCoefficient * self.velocityBias[i][j] - self.learningRate * (biasGradients[i][j] / self.batchSize)
                                self.biases[i][j] += self.velocityBias[i][j]
                            else:
                                self.biases[i][j] -= self.learningRate * (biasGradients[i][j] / self.batchSize)
                    
                    weightGradients, biasGradients = [], []
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
                        biasGradients.append(b)
            self.momentumCoefficient *= self.momentumDecay
