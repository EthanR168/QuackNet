class Optimisers:
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
