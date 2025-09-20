import numpy as np

class SGD:
    def __init__(self, forwardPropagationFunction, backwardPropagationFunction, giveInputsToBackprop = False):
        self.forwardPropagationFunction = forwardPropagationFunction
        self.backwardPropagationFunction = backwardPropagationFunction
        self.giveInputsToBackprop = giveInputsToBackprop 

    def optimiser(self, inputData, labels, useBatches, batchSize, learningRate):
        return self._trainStochasticGradientDescent_WithoutBatches(inputData, labels, learningRate)

    def _trainStochasticGradientDescent_WithoutBatches(self, inputData, labels, learningRate):
        allNodes = []   
        for data in range(len(inputData)):
            batchData = np.array([inputData[data]])
            batchLabel = np.array([labels[data]])

            layerNodes = self.forwardPropagationFunction(batchData)
            allNodes.append(layerNodes)

            if(self.giveInputsToBackprop == False):
                Parameters, Gradients = self.backwardPropagationFunction(layerNodes, batchLabel)
            else:
                Parameters, Gradients = self.backwardPropagationFunction(batchData, layerNodes, batchLabel)

            Parameters = self._updateWeightsBiases(Parameters, Gradients, learningRate)
        return allNodes, Parameters
    
    def _updateWeightsBiases(self, Parameters, Gradients, learningRate):
        for key in Gradients:
            if(isinstance(Gradients[key], list)): # if the Gradient is a inhomengous list (jagged array, which numpy doesnt like)
                for i in range(len(Gradients[key])):
                    Parameters[key][i] -= learningRate * Gradients[key][i]
            else:
                Parameters[key] -= learningRate * Gradients[key]

        return Parameters