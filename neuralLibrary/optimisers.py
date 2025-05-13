import numpy as np

class Optimisers:
    def trainGradientDescent(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, _):
        layerNodes = []
        if(useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            weightGradients, biasGradients = self.initialiseGradients(weights, biases)
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, weights, biases, labels)
                self.addGradients(weightGradients, biasGradients, w, b)
            self.updateWeightsBiases(len(inputData), weights, biases, weightGradients, biasGradients, velocityWeight, velocityBias, useMomentum, momentumCoefficient, learningRate)
            momentumCoefficient *= momentumDecay

    def trainStochasticGradientDescent(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, _):
        layerNodes = []
        if(useMomentum == True):
            self.initialiseVelocity()        
        for _ in range(epochs):
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, weights, biases, labels)
                if(useMomentum == True):
                    velocityWeight = momentumCoefficient * velocityWeight - learningRate * w
                    weights += velocityWeight
                    velocityBias = momentumCoefficient * velocityBias - learningRate * b
                    biases += velocityBias
                else:
                    weights -= learningRate * w
                    biases -= learningRate * b

            momentumCoefficient *= momentumDecay

    def trainGradientDescentUsingBatching(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, batchSize):
        layerNodes = []
        if(useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            weightGradients, biasGradients = self.initialiseGradients()
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, weights, biases, labels)
                self.addGradients(w, b)
                if((data + 1) % batchSize == 0):
                    self.updateWeightsBiases(batchSize, weights, biases, weightGradients, biasGradients, velocityWeight, velocityBias, useMomentum, momentumCoefficient, learningRate)
                    self.initialiseGradients()
            momentumCoefficient *= momentumDecay

    def initialiseVelocity(self, velocityWeight, velocityBias, weights, biases):
        if(velocityWeight == None):
            velocityWeight = []
            for i in weights:
                velocityWeight.append(np.zeros_like(i))
        if(velocityBias == None):
            velocityBias = []
            for i in biases:
                velocityBias.append(np.zeros_like(i))
        return velocityWeight, velocityBias
    
    def initialiseGradients(self, weights, biases):
        weightGradients, biasGradients = [], []
        for i in weights:
            weightGradients.append(np.zeros_like(i))
        for i in biases:
            biasGradients.append(np.zeros_like(i))
        return weightGradients, biasGradients

    def addGradients(self, weightGradients, biasGradients, w, b):
        for i in range(len(weightGradients)):
            weightGradients[i] += w[i].T
        for i in range(len(biasGradients)):
            biasGradients[i] += b[i].T
        return weightGradients, biasGradients
    
    def updateWeightsBiases(self, size, weights, biases, weightGradients, biasGradients, velocityWeight, velocityBias, useMomentum, momentumCoefficient, learningRate):
        if(useMomentum == True):
            for i in range(len(weights)):
                velocityWeight[i] -= momentumCoefficient * velocityWeight[i] - learningRate * (weightGradients[i] / size)
                weights[i] += velocityWeight[i]

            for i in range(len(biases)):
                velocityBias[i] = momentumCoefficient * velocityBias[i] - learningRate * (biasGradients[i] / size)
                biases[i] += velocityBias[i]
        else:
            for i in range(len(weights)):
                weights[i] -= learningRate * (weightGradients[i] / size)
            for i in range(len(biases)):
                biases[i] -= learningRate * (biasGradients[i] / size)
        return weights, biases, velocityWeight, velocityBias