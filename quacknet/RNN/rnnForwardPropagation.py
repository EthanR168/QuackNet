import numpy as np

"""
Forward propagation for RNN's is:
z = W1 * x + W2 * h(t - 1) + b
h(t) = a(z)  

W1        = the weights for the input data
x         = input data
W2        = the weights for the hidden state
h(t - 1)  = hidden state at timestamp (t - 1)
b         = bias
a         = is the activation function
h(t)      = hidden state at the current timestamp t
"""

class RNNForward():
    def _forwardPropagation(self, inputData, inputWeights, hiddenStatesWeights, biases, outputWeights, outputBiases):
        preActivationValues = []
        output = None
        input = np.array(inputData)
        for i in range(len(self.layers)):
            input, summ = self._forward(input, self.hiddenState[i], self.activationFunction, inputWeights[i], hiddenStatesWeights[i], biases[i])
            self.hiddenState[i] = input
            preActivationValues.append(summ)

        if(self.useOutputLayer == True):
            output, outputPreActivation = self._outputLayer(input, outputWeights, outputBiases, self.activationFunction)
            preActivationValues.append(outputPreActivation)
        return preActivationValues, output

    def _forward(self, input, oldHiddenState, activationFunction, inputWeights, hiddenStatesWeights, bias):
        summ = np.dot(inputWeights, input) + np.dot(hiddenStatesWeights, oldHiddenState) + bias
        newHiddenState = activationFunction(summ)
        return newHiddenState, summ
    
    def _outputLayer(self, hiddenState, outputWeights, outputBiases, activationFunction):
        summ = np.dot(outputWeights, hiddenState) + outputBiases
        return activationFunction(summ), summ
    