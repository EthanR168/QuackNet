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
            self.velocityWeight = []
            for i in self.weights:
                self.velocityWeight.append(np.zeros_like(i))

        if(self.velocityBias == None):
            self.velocityBias = []
            for i in self.biases:
                self.velocityBias.append(np.zeros_like(i))
    
    def initialiseGradients(self):
        self.weightGradients = []
        for i in self.weights:
            self.weightGradients.append(np.zeros_like(i))

        self.biasGradients = []
        for i in self.biases:
            self.biasGradients.append(np.zeros_like(i))

        #self.weightGradients, self.biasGradients = np.zeros_like(self.weights), np.zeros_like(self.biases)

    def addGradients(self, w, b):
        self.weightGradients = self.weightGradients + w  
        self.biasGradients = self.biasGradients + b
    
    def updateWeightsBiases(self, size):
        if(self.useMomentum == True):
            #self.velocityWeight = self.momentumCoefficient * self.velocityWeight - self.learningRate * (self.weightGradients / size)
            #self.weights += self.velocityWeight
            #self.velocityBias = self.momentumCoefficient * self.velocityBias - self.learningRate * (self.biasGradients / size)
            #self.biases += self.velocityBias

            for i in range(len(self.weights)):
                self.velocityWeight[i] -= self.momentumCoefficient * self.velocityWeight[i] - self.learningRate * (self.weightGradients[i] / size)
                self.weights[i] += self.velocityWeight[i]

            for i in range(len(self.biases)):
                self.velocityBias[i] = self.momentumCoefficient * self.velocityBias[i] - self.learningRate * (self.biasGradients[i] / size)
                self.biases[i] += self.velocityBias[i]

        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learningRate * (self.weightGradients[i] / size)

            for i in range(len(self.biases)):
                self.biases[i] -= self.learningRate * (self.biasGradients[i] / size)