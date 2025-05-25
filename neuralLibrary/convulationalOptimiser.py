import numpy as np

class CNNoptimiser:
    def AdamsOptimiserWithBatches(self, inputData, labels, weights, biases, batchSize, alpha, beta1, beta2, epsilon):
        firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias = self.initialiseMoment(weights, biases)
        weightGradients, biasGradients = self.initialiseGradients(weights, biases)
        allNodes = []
        for i in range(0, len(inputData), batchSize):
            batchData = inputData[i:i+batchSize]
            batchLabels = labels[i:i+batchSize]
            for j in range(len(batchData)):
                layerNodes = self.forward(batchData[j])
                allNodes.append(layerNodes)
                w, b = self.backpropagation(layerNodes, batchLabels[j])
                weightGradients, biasGradients = self.addGradients(weightGradients, biasGradients, w, b)
            weights, biases, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias = self.Adams(weightGradients, biasGradients, weights, biases, i + 1, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias, alpha, beta1, beta2, epsilon)
            weightGradients, biasGradients = self.initialiseGradients(weights, biases)
        return allNodes, weights, biases

    def AdamsOptimiserWithoutBatches(self, inputData, labels, weights, biases, alpha, beta1, beta2, epsilon):
        firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias = self.initialiseMoment(weights, biases)
        weightGradients, biasGradients = self.initialiseGradients(weights, biases)
        allNodes = []
        for i in range(len(inputData)):
            layerNodes = self.forward(inputData[i])
            allNodes.append(layerNodes)
            w, b = self.backpropagation(layerNodes, labels[i])
            weightGradients, biasGradients = self.addGradients(weightGradients, biasGradients, w, b)
            weights, biases, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias = self.Adams(weightGradients, biasGradients, weights, biases, i + 1, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias, alpha, beta1, beta2, epsilon)
            weightGradients, biasGradients = self.initialiseGradients(weights, biases)
        return allNodes, weights, biases

    def Adams(self, weightGradients, biasGradients, weights, biases, timeStamp, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias, alpha, beta1, beta2, epsilon):
        for i in range(len(weights)):
            firstMomentWeight[i] = beta1 * firstMomentWeight[i] + (1 - beta1) * weightGradients[i]
            secondMomentWeight[i] = beta2 * secondMomentWeight[i] + (1 - beta2) * (weightGradients[i] ** 2)

            firstMomentWeightHat = firstMomentWeight[i] / (1 - beta1 ** timeStamp)
            secondMomentWeightHat = secondMomentWeight[i] / (1 - beta2 ** timeStamp)

            weights[i] -= alpha * firstMomentWeightHat / (np.sqrt(secondMomentWeightHat) + epsilon)
        
        for i in range(len(biases)):
            firstMomentBias[i] = beta1 * firstMomentBias[i] + (1 - beta1) * biasGradients[i]
            secondMomentBias[i] = beta2 * secondMomentBias[i] + (1 - beta2) * (biasGradients[i] ** 2)

            firstMomentBiasHat = firstMomentBias[i] / (1 - beta1 ** timeStamp)
            secondMomentBiasHat = secondMomentBias[i] / (1 - beta2 ** timeStamp)

            biases[i] -= alpha * firstMomentBiasHat / (np.sqrt(secondMomentBiasHat) + epsilon)
        return weights, biases, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias

    def initialiseGradients(self, weights, biases):
        weightGradients, biasGradients = [], []
        for i in weights:
            weightGradients.append(np.zeros_like(i))
        for i in biases:
            biasGradients.append(np.zeros_like(i))
        return weightGradients, biasGradients

    def addGradients(self, weightGradients, biasGradients, w, b):
        print(np.array(weightGradients[0]).shape)
        print(np.array(weightGradients[1]).shape)
        print(np.array(w[0]).shape)
        for i in range(len(weightGradients)):
            weightGradients[i] += w[i]
            weightGradients[i] = np.clip(weightGradients[i], -1, 1)
        for i in range(len(biasGradients)):
            biasGradients[i] += b[i].T
            biasGradients[i] = np.clip(biasGradients[i], -1, 1)
        return weightGradients, biasGradients

    def initialiseMoment(self, weights, biases):
        momentWeight = []
        for i in weights:
            momentWeight.append(np.zeros_like(i))
        momentBias = []
        for i in biases:
            momentBias.append(np.zeros_like(i))
        return momentWeight, momentBias, momentWeight, momentBias

    