import numpy as np

"""
RNN backprop goes through every hidden state at every timestamp (because in forward prop the past hidden state is used to update the new one)
So a stacked RNN of 5 which went through 10 timestamps would do backprop 50 times

Anacroynm for backpropagation in RNN's is BPTT (Back Propagation Through Time) as it backprops through time

In RNN forward prop the output can either be using the last hidden state or using a output layer
output layer BPTT requires an extra step at the start compared to the last hidden state BPTT, but the rest is the same

For rnn with 1 hidden state (not stacked) BPTT goes through time
For stacked RNN with multiple hidden states BPTT goes through time and then hidden states
"""

class BPTTSingularHiddenState(): # only 1 hidden state
    def _BPTTNoOutputLayer(self, inputs, hiddenStates, preActivationValues, targets): # no output layer
        inputWeightGradients = np.zeros_like(self.inputWeights)
        hiddenStateWeightGradients = np.zeros_like(self.hiddenWeights)
        biasGradients = np.zeros_like(self.biases)

        delta = np.zeros_like(hiddenStates[0])

        for i in reversed(range(self.numberOfTimeStamps)):
            loss = self.lossDerivative(hiddenStates[i], targets[i])
            error = (loss + delta) * self.activationDerivative(preActivationValues[i])

            inputWeightGradients += np.outer(error, inputs[i])
            biasGradients += error

            if(i > 0):
                hiddenStateWeightGradients += np.outer(error, hiddenStates[i - 1])
            
            delta = error @ self.hiddenWeights
        return inputWeightGradients, hiddenStateWeightGradients, biasGradients
    
    def _BPTTWithOutputLayer(self, inputs, hiddenStates, preActivationValues, targets, outputs): # with output layer
        inputWeightGradients = np.zeros_like(self.inputWeights)
        hiddenStateWeightGradients = np.zeros_like(self.hiddenWeights)
        biasGradients = np.zeros_like(self.biases)

        outputWeightGradients = np.zeros_like(self.outputWeights)
        outputbiasGradients = np.zeros_like(self.outputBiases)

        delta = np.zeros_like(hiddenStates[0])

        for i in reversed(range(self.numberOfTimeStamps)):
            outputLoss = self.lossDerivative(outputs[i], targets[i])
            outputWeightGradients += np.outer(outputLoss, hiddenStates[i])
            outputbiasGradients += outputLoss

            error = (outputLoss @ self.outputWeights + delta) * self.activationDerivative(preActivationValues[i])

            inputWeightGradients += np.outer(error, inputs[i])
            biasGradients += error
            if(i > 0):
                hiddenStateWeightGradients += np.outer(error, hiddenStates[i - 1])
            
            delta = error @ self.hiddenWeights
        return inputWeightGradients, hiddenStateWeightGradients, biasGradients, outputWeightGradients, outputbiasGradients

class BPTTStackedRNN(): # stacked hidden states
    def _BPTTNoOutputLayer(self, inputs, hiddenStates, preActivationValues, targets): # no output layer
        inputWeightGradients = [np.zeros_like(W) for W in self.inputWeights]
        hiddenStateWeightGradients = [np.zeros_like(W) for W in self.hiddenWeights]
        biasGradients = [np.zeros_like(b) for b in self.biases]

        deltas = [np.zeros_like(hiddenStates[l][0]) for l in range(len(self.inputWeights))]

        for i in reversed(range(self.numberOfTimeStamps)):
            for l in reversed(range(len(self.inputWeights))):
                if(l == len(self.inputWeights) - 1):
                    loss = self.lossDerivative(hiddenStates[l][i], targets[i])
                    deltas[l] = loss + deltas[l]

                error = deltas[l] * self.activationDerivative(preActivationValues[l][i])

                biasGradients[l] += error
                if(l == 0):
                    inputWeightGradients[l] += np.outer(error, inputs[i])
                else:
                    inputWeightGradients[l] += np.outer(error, hiddenStates[l - 1][i])
                
                if(i > 0):
                    hiddenStateWeightGradients[l] += np.outer(error, hiddenStates[l][i - 1])
                
                deltas[l] = error @ self.hiddenWeights[l]
                if(l > 0):
                    deltas[l - 1] += error @ self.inputWeights[l]

        return inputWeightGradients, hiddenStateWeightGradients, biasGradients
    
    def _BPTTWithOutputLayer(self, inputs, hiddenStates, preActivationValues, targets, outputs): # with output layer
        inputWeightGradients = [np.zeros_like(W) for W in self.inputWeights]
        hiddenStateWeightGradients = [np.zeros_like(W) for W in self.hiddenWeights]
        biasGradients = [np.zeros_like(b) for b in self.biases]

        outputWeightGradients = np.zeros_like(self.outputWeights)
        outputbiasGradients = np.zeros_like(self.outputBiases)

        deltas = [np.zeros_like(hiddenStates[l][0]) for l in range(len(self.inputWeights))]

        for i in reversed(range(self.numberOfTimeStamps)):
            outputLoss = self.lossDerivative(outputs[i], targets[i])
            outputWeightGradients += np.outer(outputLoss, hiddenStates[-1][i])
            outputbiasGradients += outputLoss

            deltas[-1] = outputLoss @ self.outputWeights

            for l in reversed(range(len(self.inputWeights))):
                error = deltas[l] * self.activationDerivative(preActivationValues[l][i])

                biasGradients[l] += error
                if(l == 0):
                    inputWeightGradients[l] += np.outer(error, inputs[i])
                else:
                    inputWeightGradients[l] += np.outer(error, hiddenStates[l - 1][i])
                
                if(i > 0):
                    hiddenStateWeightGradients[l] += np.outer(error, hiddenStates[l][i - 1])
                
                deltas[l] = error @ self.hiddenWeights[l]
                if(l > 0):
                    deltas[l - 1] += error @ self.inputWeights[l]

        return inputWeightGradients, hiddenStateWeightGradients, biasGradients, outputWeightGradients, outputbiasGradients

