import numpy as np

class RMSProp:
    def __init__(self, forwardPropagationFunction, backwardPropagationFunction, giveInputsToBackprop=False, decay=0.9, epsilon=1e-8):
        self.squaredAvg = {}  
        self.forwardPropagationFunction = forwardPropagationFunction
        self.backwardPropagationFunction = backwardPropagationFunction
        self.giveInputsToBackprop = giveInputsToBackprop
        self.decay = decay
        self.epsilon = epsilon

    def optimiser(self, inputData, labels, useBatches, batchSize, alpha):
        if useBatches:
            return self._RMSPropOptimiserWithBatches(inputData, labels, batchSize, alpha)
        else:
            return self._RMSPropOptimiserWithoutBatches(inputData, labels, alpha)

    def _RMSPropOptimiserWithBatches(self, inputData, labels, batchSize, alpha):
        AllOutputs = []
        numBatches = len(inputData) // batchSize
        for i in range(numBatches):
            batchData = np.array(inputData[i*batchSize:(i+1)*batchSize])
            batchLabels = np.array(labels[i*batchSize:(i+1)*batchSize])
            Parameters = None

            output = self.forwardPropagationFunction(batchData)
            AllOutputs.append(output)

            if not self.giveInputsToBackprop:
                Parameters, Gradients = self.backwardPropagationFunction(output, batchLabels)
            else:
                Parameters, Gradients = self.backwardPropagationFunction(batchData, output, batchLabels)

            for key in Gradients:
                if isinstance(Gradients[key], list):
                    for j in range(len(Gradients[key])):
                        Gradients[key][j] = np.array(Gradients[key][j]) / batchSize
                else:
                    Gradients[key] = Gradients[key] / batchSize

            Parameters = self._RMSPropUpdate(Parameters, Gradients, alpha)
        return AllOutputs, Parameters

    def _RMSPropOptimiserWithoutBatches(self, inputData, labels, alpha):
        AllOutputs = []
        for i in range(len(inputData)):
            input = np.array([inputData[i]])
            lab = np.array([labels[i]])
            output = self.forwardPropagationFunction(input)
            AllOutputs.append(output)

            if not self.giveInputsToBackprop:
                Parameters, Gradients = self.backwardPropagationFunction(output, lab)
            else:
                Parameters, Gradients = self.backwardPropagationFunction(input, output, lab)

            Parameters = self._RMSPropUpdate(Parameters, Gradients, alpha)
        return AllOutputs, Parameters

    def _RMSPropUpdate(self, Parameters, Gradients, alpha):
        for key in Gradients:
            if isinstance(Gradients[key], list):
                for i, grad in enumerate(Gradients[key]):
                    grad = np.array(grad)
                    if key not in self.squaredAvg:
                        self.squaredAvg[key] = [np.zeros_like(g) for g in Gradients[key]]
                    self.squaredAvg[key][i] = self.decay * self.squaredAvg[key][i] + (1 - self.decay) * (grad ** 2)
                    Parameters[key][i] -= alpha * grad / (np.sqrt(self.squaredAvg[key][i]) + self.epsilon)
            else:
                if key not in self.squaredAvg:
                    self.squaredAvg[key] = np.zeros_like(Gradients[key])
                self.squaredAvg[key] = self.decay * self.squaredAvg[key] + (1 - self.decay) * (Gradients[key] ** 2)
                Parameters[key] -= alpha * Gradients[key] / (np.sqrt(self.squaredAvg[key]) + self.epsilon)
        return Parameters