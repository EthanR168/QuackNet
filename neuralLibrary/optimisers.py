import numpy as np

class Optimisers:
    def trainGradientDescent(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, _):
        print("eee ", weights[0][0][0])
        l = []
        if(useMomentum == True):
            self.initialiseVelocity()
        for _ in range(epochs):
            weightGradients, biasGradients = self.initialiseGradients(weights, biases)
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                l.append(layerNodes[len(layerNodes) - 1])
                w, b = self.backPropgation(layerNodes, weights, biases, labels[data])
                velocityWeight, velocityBias = self.addGradients(weightGradients, biasGradients, w, b)
            print("Sum of weight gradients:", [wg.sum() for wg in weightGradients])
            print("Sum of bias gradients:", [bg.sum() for bg in biasGradients])
            print("Shape w[0]:", w[0].shape)
            print("Sum of w[0]:", np.sum(w[0]))
            print("Some sample values w[0][:5,:5]:", w[0][:5,:5])
            weights, biases, velocityWeight, velocityBias = self.updateWeightsBiases(len(inputData), weights, biases, weightGradients, biasGradients, velocityWeight, velocityBias, useMomentum, momentumCoefficient, learningRate)
            momentumCoefficient *= momentumDecay
        print(weights[0][0][0])
        return l, weights, biases, velocityWeight, velocityBias

    def trainStochasticGradientDescent(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, _):
        l = []
        if(useMomentum == True):
            self.initialiseVelocity()        
        for _ in range(epochs):
            for data in range(len(inputData)):
                layerNodes = self.forwardPropagation(inputData[data])
                l.append(layerNodes)
                w, b = self.backPropgation(layerNodes, weights, biases, labels[data])
                if(useMomentum == True):
                    velocityWeight = momentumCoefficient * velocityWeight - learningRate * w
                    weights += velocityWeight
                    velocityBias = momentumCoefficient * velocityBias - learningRate * b
                    biases += velocityBias
                else:
                    weights -= learningRate * w
                    biases -= learningRate * b

            momentumCoefficient *= momentumDecay
        return l, weights, biases, self.velocityWeight, self.velocityBias

    def trainGradientDescentUsingBatching(self, inputData, labels, epochs, weights, biases, momentumCoefficient, momentumDecay, useMomentum, velocityWeight, velocityBias, learningRate, batchSize):
        l = []
        if(useMomentum == True):
            velocityWeight, velocityBias = self.initialiseVelocity()
        for _ in range(epochs):
            for i in range(0, len(inputData), batchSize):
                batchData = inputData[i:i+batchSize]
                batchLabels = labels[i:i+batchSize]
                weightGradients, biasGradients = self.initialiseGradients()
                for j in range(len(batchData)):
                    layerNodes = self.forwardPropagation(batchData[j])
                    l.append(layerNodes)
                    w, b = self.backPropgation(layerNodes, weights, biases, batchLabels[j])
                    weightGradients, biasGradients = self.addGradients(weightGradients, biasGradients, w, b)
                self.updateWeightsBiases(batchSize, weights, biases, weightGradients, biasGradients, velocityWeight, velocityBias, useMomentum, momentumCoefficient, learningRate)
            momentumCoefficient *= momentumDecay
        return l, weights, biases, self.velocityWeight, self.velocityBias

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
                print("aaaaaaaaaaaaaaaaaaaa")
                print(type(weights), [type(w) for w in weights])
                print("Before update weights[i][0][0]:", weights[i][0][0])
                print("size of change: ", (learningRate * (weightGradients[i] / size))[0][0])
                print("Max gradient before division:", np.max(weightGradients[i]))
                print("Min gradient before division:", np.min(weightGradients[i]))
                print("Mean gradient before division:", np.mean(weightGradients[i]))
                print("Raw gradient value:", weightGradients[i][0][0])
                print("size ", size)
                print(f"Layer {i} weight shape: {weights[i].shape}, gradient shape: {weightGradients[i].shape}")
                weights[i] = weights[i] - learningRate * (weightGradients[i] / size)
                print("After update weights[i][0][0]:", weights[i][0][0])
                idx = (10, 10)  # example index
                print("Raw gradient at index", idx, ":", weightGradients[i][idx])
                print("Change at index", idx, ":", learningRate * (weightGradients[i][idx] / size))
                print("Before update weights[i][idx]:", weights[i][idx])
                weights[i][idx] -= learningRate * (weightGradients[i][idx] / size)
                print("After update weights[i][idx]:", weights[i][idx])
                print("Shape weightGradients[i]:", weightGradients[i].shape)
                import time
                time.sleep(60)
            for i in range(len(biases)):
                biases[i] -= learningRate * (biasGradients[i] / size)
        return weights, biases, velocityWeight, velocityBias