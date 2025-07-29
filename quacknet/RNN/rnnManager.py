from quacknet.core.activationFunctions import relu, sigmoid, tanH, linear, softMax
from quacknet.core.activationDerivativeFunctions import ReLUDerivative, SigmoidDerivative, TanHDerivative, LinearDerivative, SoftMaxDerivative
from quacknet.core.lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction
from quacknet.core.lossDerivativeFunctions import MSEDerivative, MAEDerivative, CrossEntropyLossDerivative
from quacknet.RNN.rnnForwardPropagation import RNNForward
from quacknet.RNN.rnnBackPropagation import BPTTSingularHiddenState, BPTTStackedRNN
from quacknet.RNN.rnnCreateWeights import RNNInitialiser
from quacknet.RNN.rnnOptimiser import RNNOptimiser
import numpy as np

"""
RNN has 1 hidden state, which used the input data and the hidden state from the time step before it to calculate its new value
To make a RNN with multiple hidden states you stack them, where the output of a hidden state is the input to another hidden state

For sake of simplicity you can choose to stack the RNN in the class itself, instead of creating lots of differant classes
"""

class RNN(RNNForward, BPTTSingularHiddenState, BPTTStackedRNN, RNNInitialiser, RNNOptimiser):
    def __init__(self, numberOfTimeStamps, useOutputLayer = True, SizeOfHiddenStates = 1, activationFunction = "relu", outputActivationFunction = "relu", lossFunc = "MSE", stackRNN = False, NumberOfHiddenStates = 1):
        self.useOutputLayer = useOutputLayer # whether to use a output layer as output or use the hidden state as output
        self.numberOfTimeStamps = numberOfTimeStamps
        self.stackRNN = stackRNN
        self.NumberOfHiddenStates = NumberOfHiddenStates 
        self.SizeOfHiddenStates = SizeOfHiddenStates
        self.layers = np.zeros(self.NumberOfHiddenStates)

        funcs = {
            "relu": relu,
            "sigmoid": sigmoid,
            "linear": linear,
            "tanh": tanH,
            "softmax": softMax,
        }
        if(activationFunction.lower() not in funcs):
            raise ValueError(f"Activation function not made: {activationFunction.lower()}")
        self.activationFunction = funcs[activationFunction.lower()]

        if(outputActivationFunction.lower() not in funcs):
            raise ValueError(f"Activation function not made: {outputActivationFunction.lower()}")
        self.outputActivationFunction = funcs[outputActivationFunction.lower()]

        derivs = {
            relu: ReLUDerivative,
            sigmoid: SigmoidDerivative,
            linear: LinearDerivative,
            tanH: TanHDerivative,
            softMax: SoftMaxDerivative,
        }
        self.activationDerivative = derivs[self.activationFunction]
        self.outputLayerDerivative = derivs[self.outputActivationFunction]

        lossFunctionDict = {
            "mse": MSELossFunction,
            "mae": MAELossFunction,
            "cross entropy": CrossEntropyLossFunction,"cross": CrossEntropyLossFunction,
        }
        self.lossFunction = lossFunctionDict[lossFunc.lower()]

        lossDerivs = {
            MSELossFunction: MSEDerivative,
            MAELossFunction: MAEDerivative,
            CrossEntropyLossFunction: CrossEntropyLossDerivative,
        }
        self.lossDerivative = lossDerivs[self.lossFunction]

        self.outputs = []
        self.hiddenState = []
        self.allHiddenStates = [] # list of lists of the hidden states of the whole RNN network (if not stacked then size will be 1)
        self.preActivationValues = [] # list of lists of the preactivated values gotten during forward propagation

        self.outputWeights = []
        self.outputBiases = []
        self.inputWeights = []
        self.hiddenStatesWeights = []
        self.biases = []

    def RNNForwardPropagation(self, inputData):
        self.allHiddenStates = [] # list of every hidden state at every timestamp
        self.outputs = []
        self.preActivationValues = []
        self.hiddenState = [np.zeros_like(self.biases[i]) for i in range(len(self.layers))]
        for t in range(self.numberOfTimeStamps):
            currentInput = inputData[t]
            preActivation, output = self._forwardPropagation(currentInput, self.inputWeights, self.hiddenStatesWeights, self.biases, self.outputWeights, self.outputBiases)
            self.allHiddenStates.append([h.copy() for h in self.hiddenState])
            self.preActivationValues.append(preActivation)
            self.outputs.append(output)

    def RNNBackwardPropagation(self, inputs, targets): #self, inputs, hiddenStates, preActivationValues, targets, outputs
        if(self.stackRNN == True):
            BPTT = BPTTStackedRNN()
            if(self.useOutputLayer == True):
                return BPTT._Stacked_BPTTWithOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets, self.outputs)
            else:
                return BPTT._Stacked_BPTTNoOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets)
        else:
            BPTT = BPTTSingularHiddenState()
            if(self.useOutputLayer == True):
                return BPTT._Singular_BPTTWithOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets, self.outputs)
            else:
                return BPTT._Singular_BPTTNoOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets)

    def CreateWeightsBiases(self, inputSize, outputSize = None):
        self._CreateWeightsBiases(inputSize, outputSize)