from quacknet.core.activationFunctions import relu, sigmoid, tanH, linear, softMax
from quacknet.core.activationDerivativeFunctions import ReLUDerivative, SigmoidDerivative, TanHDerivative, LinearDerivative, SoftMaxDerivative
from quacknet.core.lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction
from quacknet.core.lossDerivativeFunctions import MSEDerivative, MAEDerivative, CrossEntropyLossDerivative
from quacknet.RNN.rnnForwardPropagation import RNNForward
from quacknet.RNN.rnnBackPropagation import BPTTSingularHiddenState, BPTTStackedRNN

"""
RNN has 1 hidden state, which used the input data and the hidden state from the time step before it to calculate its new value
To make a RNN with multiple hidden states you stack them, where the output of a hidden state is the input to another hidden state

For sake of simplicity you can choose to stack the RNN in the class itself, instead of creating lots of differant classes
"""

class RNN(RNNForward, BPTTSingularHiddenState, BPTTStackedRNN):
    def __init__(self, numberOfTimeStamps, useOutputLayer = True, SizeOfHiddenStates = 1, activationFunction = "relu", lossFunc = "MSE", stackRNN = False, NumberOfHiddenStates = 1):
        self.useOutputLayer = useOutputLayer # whether to use a output layer as output or use the hidden state as output
        self.numberOfTimeStamps = numberOfTimeStamps
        self.stackRNN = stackRNN
        self.NumberOfHiddenStates = NumberOfHiddenStates 
        self.SizeOfHiddenStates = SizeOfHiddenStates
        self.layers = []

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

        derivs = {
            relu: ReLUDerivative,
            sigmoid: SigmoidDerivative,
            linear: LinearDerivative,
            tanH: TanHDerivative,
            softMax: SoftMaxDerivative,
        }
        self.activationDerivative = derivs[activationFunction]

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

    def RNNForwardPropagation(self, inputData, inputWeights, hiddenStatesWeights, biases):
        self.allHiddenStates = [] # list of every hidden state at every timestamp
        for _ in range(self.numberOfTimeStamps):
            preActivation, output = self._forwardPropagation(inputData, inputWeights, hiddenStatesWeights, biases)
            self.allHiddenStates.append(self.HiddenStates)
            self.preActivationValues.append(preActivation)
            self.outputs.append(output)

    def RNNBackwardPropagation(self, inputs, targets): #self, inputs, hiddenStates, preActivationValues, targets, outputs
        if(self.stackRNN == True):
            BPTT = BPTTStackedRNN()
            if(self.useOutputLayer == True):
                BPTT._BPTTWithOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets, self.output)
            else:
                BPTT._BPTTNoOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets)
        else:
            BPTT = BPTTSingularHiddenState()
            if(self.useOutputLayer == True):
                BPTT._BPTTWithOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets, self.output)
            else:
                BPTT._BPTTNoOutputLayer(inputs, self.allHiddenStates, self.preActivationValues, targets)