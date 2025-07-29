from quacknet.core.activationFunctions import relu, sigmoid
import numpy as np
import math

class RNNInitialiser():
    def _CreateWeightsBiases(self, inputSize, outputSize):
        for i in range(len(self.NumberOfHiddenStates)):
            self.inputWeights.append(self._initialiseWeights(self.NumberOfHiddenStates[i], inputSize))
            self.hiddenStatesWeights.append(self._initialiseWeights(self.NumberOfHiddenStates[i], self.NumberOfHiddenStates[i]))
            self.biases.append(np.zeros(self.NumberOfHiddenStates[i]))

            inputSize = self.NumberOfHiddenStates[i]

        if(self.useOutputLayer == True):
            self.outputWeights = self._initialiseWeights(outputSize, self.NumberOfHiddenStates[-1])
            self.outputBiases = np.zeros(outputSize)

    def _initialiseWeights(self, outputSize, inputSize):
        actFunc = self.activationFunction

        if(actFunc == relu):
            bounds = math.sqrt(2 / inputSize) # He initialisation
        elif(actFunc == sigmoid):
            bounds = math.sqrt(6 / (inputSize + outputSize)) # Xavier initialisation
        else:
            bounds = 1
            
        w = np.random.normal(0, bounds, size=(outputSize, inputSize))
        return w