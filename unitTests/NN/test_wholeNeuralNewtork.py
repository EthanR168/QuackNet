import numpy as np
from neuralLibrary.activationFunctions import relu, linear
from neuralLibrary.backPropgation import backPropgation
from neuralLibrary.activationDerivativeFunctions import ReLUDerivative, LinearDerivative
from neuralLibrary.lossDerivativeFunctions import MSEDerivative
from neuralLibrary.lossFunctions import MSELossFunction
from neuralLibrary.main import Network

np.random.seed(55)

def createData():
    inputData = np.random.rand(5)
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
    return inputData, weights, biases, trueValues, layers

def calculateNodes(input, weights, biases):
    layerNodes = [input]
    layerNodes.append(np.dot(layerNodes[0], weights[0]) + biases[0])
    layerNodes.append(np.dot(layerNodes[1], weights[1]) + biases[1])
    return layerNodes

def forwardPropagation(inputData, weights, biases, layers):
    n = Network()
    n.layers = layers
    n.weights = weights
    n.biases = biases

    layerNodes = n.forwardPropagation(inputData)

    return layerNodes

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

def backpropagation(layerNodes, weights, biases, trueValues, layers):
    activationDerivatives = [ReLUDerivative, ReLUDerivative, LinearDerivative]
    lossDerivative = MSEDerivative

    weightGradients, biasGradients = backPropgation(layerNodes, weights, biases, trueValues, layers, MSELossFunction)

    expectedWeightGradients, expectedBiasGradients = calculateGradients(layerNodes, weights, trueValues, activationDerivatives, lossDerivative)

    return expectedWeightGradients, expectedBiasGradients, weightGradients, biasGradients

def test():
    inputData, weights, biases, trueValues, layers = createData()

    predictedLayerNodes = forwardPropagation(inputData, weights, biases, layers)
    expectedLayerNodes = calculateNodes(inputData, weights, biases)

    assert np.allclose(predictedLayerNodes[0].shape, expectedLayerNodes[0].shape)
    assert np.allclose(predictedLayerNodes[1].shape, expectedLayerNodes[1].shape)
    assert np.allclose(predictedLayerNodes[2].shape, expectedLayerNodes[2].shape)

    assert np.allclose(predictedLayerNodes[0], expectedLayerNodes[0])
    assert np.allclose(predictedLayerNodes[1], expectedLayerNodes[1])
    assert np.allclose(predictedLayerNodes[2], expectedLayerNodes[2])

    expectedWeightGradients, expectedBiasGradients, weightGradients, biasGradients = backpropagation(expectedLayerNodes, weights, biases, trueValues, layers)

    assert np.allclose(weightGradients[0].shape, expectedWeightGradients[1].shape)
    assert np.allclose(weightGradients[1].shape, expectedWeightGradients[0].shape)
    assert np.allclose(biasGradients[0].shape, expectedBiasGradients[1].shape)
    assert np.allclose(biasGradients[1].shape, expectedBiasGradients[0].shape)

    assert np.allclose(weightGradients[0], expectedWeightGradients[1])
    assert np.allclose(weightGradients[1], expectedWeightGradients[0])
    assert np.allclose(biasGradients[0], expectedBiasGradients[1])
    assert np.allclose(biasGradients[1], expectedBiasGradients[0])