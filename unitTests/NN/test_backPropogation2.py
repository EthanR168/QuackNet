import numpy as np
from quacknet.core.activationFunctions import relu, linear
from quacknet.NN.backPropgation import _backPropgation
from quacknet.core.activationDerivativeFunctions import ReLUDerivative, LinearDerivative
from quacknet.core.lossDerivativeFunctions import MSEDerivative
from quacknet.core.lossFunctions import MSELossFunction

np.random.seed(55)

def createData():
    layerNodes = [
        np.random.rand(5),
        np.random.rand(4),
        np.random.rand(3),
    ]

    weights = [
        np.random.rand(5, 4),
        np.random.rand(4, 3),
    ]

    biases = [np.random.rand(4), np.random.rand(3)]
    trueValues = np.array([0.1, 0.5, 0.9])
    layers = [
        [5, relu],
        [4, relu],
        [3, linear]
    ]
    return layerNodes, weights, biases, trueValues, layers

def calculateGradients(layerNodes, weights, trueValues, activationDerivatives, lossDerivative):
    output = layerNodes[-1]
    hidden = layerNodes[-2]
    errorTerms = lossDerivative(output, trueValues, len(output) * activationDerivatives[2](output))
    outputWeightGradients = np.outer(hidden, errorTerms)
    outputBiasGradients = errorTerms

    input = layerNodes[0]
    hiddenErrorTerms = (errorTerms @ weights[-1].T) * activationDerivatives[1](layerNodes[1])
    hiddenWeightGradients = np.outer(input, hiddenErrorTerms)
    hiddenBiasGradients = hiddenErrorTerms

    return [outputWeightGradients, hiddenWeightGradients], [outputBiasGradients, hiddenBiasGradients]

def test_backpropagation():
    layerNodes, weights, biases, trueValues, layers = createData()
    activationDerivatives = [ReLUDerivative, ReLUDerivative, LinearDerivative]
    lossDerivative = MSEDerivative

    weightGradients, biasGradients = _backPropgation(layerNodes, weights, biases, trueValues, layers, MSELossFunction)

    expectedWeightGradients, expectedBiasGradients = calculateGradients(layerNodes, weights, trueValues, activationDerivatives, lossDerivative)

    assert np.allclose(weightGradients[0].shape, expectedWeightGradients[1].shape)
    assert np.allclose(weightGradients[1].shape, expectedWeightGradients[0].shape)
    assert np.allclose(biasGradients[0].shape, expectedBiasGradients[1].shape)
    assert np.allclose(biasGradients[1].shape, expectedBiasGradients[0].shape)

    assert np.allclose(weightGradients[0], expectedWeightGradients[1])
    assert np.allclose(weightGradients[1], expectedWeightGradients[0])
    assert np.allclose(biasGradients[0], expectedBiasGradients[1])
    assert np.allclose(biasGradients[1], expectedBiasGradients[0])