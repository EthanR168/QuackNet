import numpy as np

class GD:
    def __init__(self, forwardPropagationFunction, backwardPropagationFunction, giveInputsToBackprop = False):
        self.forwardPropagationFunction = forwardPropagationFunction
        self.backwardPropagationFunction = backwardPropagationFunction
        self.giveInputsToBackprop = giveInputsToBackprop 

    def optimiser(self, inputData, labels, useBatches, batchSize, learningRate):
        return self._trainGradientDescent_Batching(inputData, labels, batchSize, learningRate)

    def _trainGradientDescent_Batching(self, inputData, labels, batchSize, learningRate):     
        AllOutputs = []
        numBatches = len(inputData) // batchSize
        import time
        for i in range(numBatches):
            start = time.time()
            batchData = np.array(inputData[i*batchSize:(i+1)*batchSize])
            batchLabels = np.array(labels[i*batchSize:(i+1)*batchSize])
            Parameters = None

            output = self.forwardPropagationFunction(batchData)
            AllOutputs.append(output)

            if(self.giveInputsToBackprop == False):
                Parameters, Gradients = self.backwardPropagationFunction(output, batchLabels)
            else:
                Parameters, Gradients = self.backwardPropagationFunction(batchData, output, batchLabels)

            for key in Gradients:
                if(isinstance(Gradients[key], list)): # inhomengous array
                    for j in range(len(Gradients[key])):
                        Gradients[key][j] = np.array(Gradients[key][j]) / batchSize
                else:
                    Gradients[key] = Gradients[key] / batchSize

            Parameters = self._updateWeightsBiases(Parameters, Gradients, learningRate)
            print(f"epoch: {i+1} / {numBatches}, took {time.time() - start}")
        return AllOutputs, Parameters
    
    def _updateWeightsBiases(self, Parameters, Gradients, learningRate):
        for key in Gradients:
            if(isinstance(Gradients[key], list)): # if the Gradient is a inhomengous list (jagged array, which numpy doesnt like)
                for i in range(len(Gradients[key])):
                    Parameters[key][i] -= learningRate * Gradients[key][i]
            else:
                Parameters[key] -= learningRate * Gradients[key]
        return Parameters