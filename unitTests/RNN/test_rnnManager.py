from quacknet.RNN.rnnManager import RNN
from quacknet.core.activationFunctions import relu
import numpy as np


def test_RNN_ForwardPropagation():
    rnn = RNN(3, False, 2, "relu", "relu", "mse", False)

    inputData = np.array([
        [1.0, 2.0],
        [0.5, -1.0],
        [2.0, 0.0]
    ])

    inputWeights = [np.array([
        [0.1, 0.2],
        [0.3, 0.4]
    ])]

    hiddenWeights = [np.array([
        [0.5, -0.5],
        [0.6, -0.6]
    ])]

    biases = [np.array([0.1, 0.2])]

    rnn.inputWeights = inputWeights 
    rnn.hiddenStatesWeights = hiddenWeights
    rnn.biases = biases
    rnn.outputWeights = None
    rnn.outputBiases =  None

    rnn.RNNForwardPropagation(inputData)
    
    assert len(rnn.allHiddenStates) == 3

    oldHidden = np.zeros(2)
    
    for t in range(3):
        expectedHidden = relu(np.dot(inputWeights[0], inputData[t]) + np.dot(hiddenWeights[0], oldHidden) + biases[0])
        assert np.allclose(rnn.allHiddenStates[t][0], expectedHidden)
        
        oldHidden = expectedHidden 