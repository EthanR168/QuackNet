from quacknet.RNN.rnnForwardPropagation import RNNForward
from quacknet.core.activationFunctions import relu
import numpy as np

def test_forward():
    input = np.array([0.5, -0.3, 0.1])
    oldHidden = np.array([0.0, 0.0])
    inputWeights = np.array([
        [0.2, -0.1, 0.4],
        [0.7, 0.3, -0.5]
    ])
    hiddenStatesWeights = np.array([
        [0.5, -0.2],
        [-0.3, 0.8]
    ])
    bias = np.array([0.1, -0.1])

    rnnForward = RNNForward()
    newHidden, summ = rnnForward._forward(input, oldHidden, relu, inputWeights, hiddenStatesWeights, bias)

    assert newHidden.shape == bias.shape
    assert summ.shape == bias.shape

    expectedSumm = np.dot(inputWeights, input) + np.dot(hiddenStatesWeights, oldHidden) + bias

    assert np.allclose(summ, expectedSumm)

    expectedHidden = relu(expectedSumm)
    assert np.allclose(newHidden, expectedHidden)

def test_outputLayer():
    hiddenState = np.array([0.3, -0.2])
    outputWeights = np.array([[0.5, -0.1], [0.4, 0.7]])
    outputBiases = np.array([0.1, -0.2])

    rnnForward = RNNForward()
    activatedOutput, summ = rnnForward._outputLayer(hiddenState, outputWeights, outputBiases, relu)

    assert activatedOutput.shape == outputBiases.shape
    assert summ.shape == outputBiases.shape

    expectedSumm = np.dot(outputWeights, hiddenState) + outputBiases

    assert np.allclose(summ, expectedSumm)

    expectedActivated = relu(expectedSumm)
    assert np.allclose(activatedOutput, expectedActivated)

def test_forwardPropagation_SingularRNN():
    inputData = np.array([0.5, -0.1])
    inputWeights = [np.array([[0.2, 0.4], [-0.3, 0.1]])]
    hiddenStateWeights = [np.array([[0.5, -0.2], [0.1, 0.3]])]
    biases = [np.array([0.1, -0.1])]

    outputWeights = np.array([0.4, -0.5])
    outputBiases = np.array([0.0])

    rnn = RNNForward()
    rnn.layers = [1]
    rnn.hiddenState = [np.array([0.1, 0.2])]
    rnn.useOutputLayer = True
    rnn.activationFunction = relu
    rnn.outputActivationFunction = relu
    preActivationValues, output = rnn._forwardPropagation(inputData, inputWeights, hiddenStateWeights, biases, outputWeights, outputBiases)

    assert len(preActivationValues) == 2

    assert preActivationValues[0].shape == (2,)
    assert preActivationValues[1].shape == (1,)
    assert output.shape == (1,)

    assert np.allclose(rnn.hiddenState[0], relu(np.dot(inputWeights[0], inputData) + np.dot(hiddenStateWeights[0], np.array([0.1, 0.2])) + biases[0]))

    expectedHidden = rnn.hiddenState[0]
    expectedOutputSumm = np.dot(outputWeights, expectedHidden) + outputBiases
    expectedOutput = relu(expectedOutputSumm)
    
    assert np.allclose(output, expectedOutput)

def test_forwardPropagation_StackedRNN():
    inputData = np.array([0.5, -0.3])
    inputWeights = [
        np.array([[0.1, -0.2], [0.4, 0.3]]),
        np.array([[0.2, 0.5], [-0.3, 0.1]])
    
    ]
    hiddenStateWeights = [
        np.array([[0.3, 0.0], [0.0, 0.3]]),
        np.array([[0.2, -0.4], [0.1, 0.6]])
    ]
    biases = [
        np.array([0.0, 0.1]),
        np.array([0.05, -0.05])
    ]

    outputWeights = np.array([0.3, -0.6])
    outputBiases = np.array([0.1])

    rnn = RNNForward()
    rnn.layers = [0, 1]
    rnn.hiddenState = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0])
    ]
    rnn.useOutputLayer = True
    rnn.activationFunction = relu
    rnn.outputActivationFunction = relu
    preActivationValues, output = rnn._forwardPropagation(inputData, inputWeights, hiddenStateWeights, biases, outputWeights, outputBiases)

    assert len(preActivationValues) == 3

    assert preActivationValues[0].shape == (2,)
    assert preActivationValues[1].shape == (2,)
    assert preActivationValues[2].shape == (1,)
    assert output.shape == (1,)

    expectedPreVals0 = np.dot(inputWeights[0], inputData) + biases[0]
    expectedHidden0 = relu(expectedPreVals0)

    expectedPreVals1 = np.dot(inputWeights[1], expectedHidden0) + biases[1]
    expectedHidden1 = relu(expectedPreVals1)

    expectedOutputPreVals = np.dot(outputWeights, expectedHidden1) + outputBiases
    expectedOutput = relu(expectedOutputPreVals)

    assert np.allclose(preActivationValues[0], expectedPreVals0)
    assert np.allclose(preActivationValues[1], expectedPreVals1)
    assert np.allclose(preActivationValues[2], expectedOutputPreVals)
    assert np.allclose(output, expectedOutput)
    

