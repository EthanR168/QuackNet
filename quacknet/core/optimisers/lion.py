import numpy as np

class Lion:
    def __init__(self, forward, backward, giveInputsToBackprop=False, beta1=0.9):
        self.forward = forward
        self.backward = backward
        self.giveInputsToBackprop = giveInputsToBackprop
        self.beta1 = beta1
        self.momentum = {}  

    def optimiser(self, inputData, labels, useBatches=True, batchSize=32, alpha=0.001):
        if useBatches:
            return self._optimiserWithBatches(inputData, labels, batchSize, alpha)
        else:
            return self._optimiserWithoutBatches(inputData, labels, alpha)

    def _optimiserWithBatches(self, inputData, labels, batchSize, alpha):
        allOutputs = []
        numBatches = len(inputData) // batchSize
        Parameters = None

        for i in range(numBatches):
            batchData = np.array(inputData[i*batchSize:(i+1)*batchSize])
            batchLabels = np.array(labels[i*batchSize:(i+1)*batchSize])

            output = self.forward(batchData)
            allOutputs.append(output)

            if not self.giveInputsToBackprop:
                Parameters, Gradients = self.backward(output, batchLabels)
            else:
                Parameters, Gradients = self.backward(batchData, output, batchLabels)

            for key in Gradients:
                if isinstance(Gradients[key], list):
                    for j in range(len(Gradients[key])):
                        Gradients[key][j] = np.array(Gradients[key][j]) / batchSize
                else:
                    Gradients[key] = Gradients[key] / batchSize

            Parameters = self._lionUpdate(Parameters, Gradients, alpha)

        return allOutputs, Parameters

    def _optimiserWithoutBatches(self, inputData, labels, alpha):
        allOutputs = []
        Parameters = None
        for i in range(len(inputData)):
            inp = np.array([inputData[i]])
            lab = np.array([labels[i]])

            output = self.forward(inp)
            allOutputs.append(output)

            if not self.giveInputsToBackprop:
                Parameters, Gradients = self.backward(output, lab)
            else:
                Parameters, Gradients = self.backward(inp, output, lab)

            Parameters = self._lionUpdate(Parameters, Gradients, alpha)
        return allOutputs, Parameters

    def _lionUpdate(self, Parameters, Gradients, alpha):
        for key in Gradients:
            if isinstance(Gradients[key], list):
                if key not in self.momentum:
                    self.momentum[key] = [np.zeros_like(g) for g in Gradients[key]]
                for i, grad in enumerate(Gradients[key]):
                    grad = np.array(grad)
                    self.momentum[key][i] = self.beta1 * self.momentum[key][i] + (1 - self.beta1) * grad
                    Parameters[key][i] -= alpha * np.sign(self.momentum[key][i])
            else:
                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(Gradients[key])
                self.momentum[key] = self.beta1 * self.momentum[key] + (1 - self.beta1) * Gradients[key]
                Parameters[key] -= alpha * np.sign(self.momentum[key])
        return Parameters
