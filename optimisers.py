class Optimisers:
    def trainGradientDescent(self, inputData, labels, epochs):
        if(self.useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            self.initialiseGradients()
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                w, b = self.backPropgation(layerNodes, self.weights, self.biases, labels)
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
            self.velocityWeight = self.initialiseListOfZeroesForWeights
        if(self.velocityBias == None):
            self.velocityBias = self.initialiseListOfZeroesForBiases
    
    def initialiseGradients(self):
        self.weightGradients, self.biasGradients = self.initialiseListOfZeroesForWeights(self.weights), self.initialiseListOfZeroesForBiases(self.biases)

    def addGradients(self, w, b):
        for i in range(len(w)):
            for j in range(len(w[i])):
                for a in range(len(w[i][j])):
                    self.weightGradients[i][j][a] += w[i][j][a] 
        for i in range(len(b)):
            for j in range(len(b[i])):
                self.biasGradients[i][j] += b[i][j]
    
    def updateWeightsBiases(self, size):
        for i in range(len(self.weightGradients)):
            for j in range(len(self.weightGradients[i])):
                for a in range(len(self.weightGradients[i][j])):
                    if(self.useMomentum == True):
                        self.velocityWeight[i][j][a] = self.momentumCoefficient * self.velocityWeight[i][j][a] - self.learningRate * (self.weightGradients[i][j][a] / self.batchSize)
                        self.weights[i][j][a] += self.velocityWeight[i][j][a]
                    else:
                        self.weights[i][j][a] -= self.learningRate * (self.weightGradients[i][j][a] / size)            
        for i in range(len(self.biasGradients)):
            for j in range(len(self.biasGradients[i])):
                if(self.useMomentum == True):
                    self.velocityBias[i][j] = self.momentumCoefficient * self.velocityBias[i][j] - self.learningRate * (self.biasGradients[i][j] / self.batchSize)
                    self.biases[i][j] += self.velocityBias[i][j]
                else:
                    self.biases[i][j] -= self.learningRate * (self.biasGradients[i][j] / size)

    def initialiseListOfZeroesForWeights(self, copy):
        new = []
        for i in range(len(copy)):
            nn = []
            for j in range(len(copy[i])):
                n = []
                for a in range(len(copy[i][j])):
                    n.append(0)
                nn.append(n)
            new.append(nn)  
        return new

    def initialiseListOfZeroesForBiases(self, copy):
        new = []
        for i in range(len(copy)):
            n = []
            for j in range(len(copy[i])):
                n.append(0)
            new.append(n)
        return new
            