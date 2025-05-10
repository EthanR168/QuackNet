import numpy as np

class Optimisers:
    def trainGradientDescent(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            self.initialiseGradients()
            for data in range(len(inputData)):
                self.layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(self.layerNodes, self.weights, self.biases, labels)
                self.addGradients(w, b)
            self.updateWeightsBiases(len(inputData))
            self.momentumCoefficient *= self.momentumDecay

    def trainStochasticGradientDescent(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            self.initialiseVelocity()        
        for _ in range(epochs):
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)
                if(self.useMomentum == True):
                    self.velocityWeight = self.momentumCoefficient * self.velocityWeight - self.learningRate * w
                    self.weights += self.velocityWeight
                    self.velocityBias = self.momentumCoefficient * self.velocityBias - self.learningRate * b
                    self.biases += self.velocityBias
                else:
                    self.weights -= self.learningRate * w
                    self.biases -= self.learningRate * b

            self.momentumCoefficient *= self.momentumDecay

    def trainGradientDescentUsingBatching(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            self.initialiseGradients()
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)
                self.addGradients(w, b)
                if((data + 1) % self.batchSize == 0):
                    self.updateWeightsBiases(self.batchSize)
                    self.initialiseGradients()
            self.momentumCoefficient *= self.momentumDecay

    def initialiseVelocity(self):
        if(self.velocityWeight == None):
            self.velocityWeight = np.zeros_like(self.weights)
        if(self.velocityBias == None):
            self.velocityBias = np.zeros_like(self.biases)
    
    def initialiseGradients(self):
        self.weightGradients = []
        for i in self.weights:
            ww = []
            for j in i:
                w = []
                for a in j:
                    w.append(0)
                ww.append(w)
            self.weightGradients.append(ww)

        self.biasGradients = []
        for i in self.biases:
            self.biasGradients.append(np.zeros_like(i))

        #self.weightGradients, self.biasGradients = np.zeros_like(self.weights), np.zeros_like(self.biases)

    def addGradients(self, w, b):
        self.weightGradients = self.weightGradients + w  
        self.biasGradients = self.biasGradients + b
    
    def updateWeightsBiases(self, size):
        if(self.useMomentum == True):
            self.velocityWeight = self.momentumCoefficient * self.velocityWeight - self.learningRate * (self.weightGradients / size)
            self.weights += self.velocityWeight
            self.velocityBias = self.momentumCoefficient * self.velocityBias - self.learningRate * (self.biasGradients / size)
            self.biases += self.velocityBias
        else:
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for a in range(len(self.weights[i][j])):
                        self.weights[i][j][a] -= self.learningRate * (self.weightGradients[i][j][a] / size)

            for i in range(len(self.biases)):
                for j in range(len(self.biases[i])):
                    self.biases[i][j] -= self.learningRate * (self.biasGradients[i][j] / size)

            #self.weights -= self.learningRate * (self.weightGradients / size)
            #self.biases -= self.learningRate * (self.biasGradients / size)
