from quacknet import Conv1DLayer, Conv2DLayer, PoolingLayer, ActivationLayer, DenseLayer, GlobalAveragePooling
from quacknet import relu, linear
from quacknet import ReLUDerivative, LinearDerivative
from quacknet import MSEDerivative
from quacknet import MSELossFunction
from quacknet import Network
import numpy as np

def test_ConvulutionalBackpropagation():
    stride = 1
    conv = Conv2DLayer(2, 1, 1, stride)

    inputTensor = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    conv.kernalWeights = np.array([[[[1, 0], [0, -1]]]])
    errorPatch = np.array([[[[1, 2], [3, 4]]]])

    weightGradients, biasGradients, errorTerms = conv._backpropagation(errorPatch, inputTensor)

    expectedWeightGradients = np.array([[[[37, 47], [67, 77]]]])
    expectedBiasGradients = np.array([10])
    expectedInputErrorTerms = np.array([[[-1, -2, 0], [-3, -3, 2], [0, 3, 4]]])

    assert expectedWeightGradients.shape == weightGradients.shape
    assert expectedBiasGradients.shape == biasGradients.shape

    assert np.allclose(weightGradients, expectedWeightGradients)
    assert np.allclose(biasGradients, expectedBiasGradients)
    assert np.allclose(errorTerms, expectedInputErrorTerms)

def test_Conv1DBackpropagation():
    stride = 1
    conv = Conv1DLayer(kernalSize=2, numKernals=1, depth=1, stride=stride, padding="no")

    inputTensor = np.array([[[1, 2, 3, 4]]])
    conv.kernalWeights = np.array([[[1, 0]]])
    errorPatch = np.array([[[1, 2, 3]]])

    weightGradients, biasGradients, errorTerms = conv._backpropagation(errorPatch, inputTensor)

    expectedWeightGradients = np.array([[[14, 20]]])
    expectedBiasGradients = np.array([6])
    expectedInputErrorTerms = np.array([[[0, 1, 2, 3]]])

    assert expectedWeightGradients.shape == weightGradients.shape
    assert expectedBiasGradients.shape == biasGradients.shape
    assert expectedInputErrorTerms.shape == errorTerms.shape

    assert np.allclose(weightGradients, expectedWeightGradients)
    assert np.allclose(biasGradients, expectedBiasGradients)
    assert np.allclose(errorTerms, expectedInputErrorTerms)

def test_MaxPoolingBackpropagation():
    inputTensor = np.array([[[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]]])
    errorPatch = np.array([[[[10, 20], [30, 40]]]])
    gridSize = 2
    strideLength = 2
    pool = PoolingLayer(gridSize, strideLength, "max")

    errorTerm = pool._backpropagation(errorPatch, inputTensor)

    expectedInputErrorTerms = np.array([[[
        [0, 0, 0, 20],
        [0, 10, 0, 0],
        [30, 0, 0, 0],
        [0, 0, 40, 0],
    ]]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_MaxPoolingBackpropagation_with3Ddim():
    inputTensor = np.array([[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]])  

    errorPatch = np.array([[
        [10, 20],
        [30, 40]
    ]]) 

    gridSize = strideLength = 2
    pool = PoolingLayer(gridSize, strideLength, "max")

    errorTerm = pool._backpropagation(errorPatch, inputTensor)

    expectedInputErrorTerms = np.array([[
        [0, 0, 0, 20],
        [0, 10, 0, 0],
        [30, 0, 0, 0],
        [0, 0, 40, 0],
    ]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_AveragePoolingBackpropagation():
    inputTensor = np.array([[[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]]])
    errorPatch = np.array([[[[10, 20], [30, 40]]]])
    gridSize = 2
    strideLength = 2
    pool = PoolingLayer(gridSize, strideLength, "ave")

    errorTerm = pool._backpropagation(errorPatch, inputTensor)

    expectedInputErrorTerms = np.array([[[
        [2.5, 2.5, 5, 5],
        [2.5, 2.5, 5, 5],
        [7.5, 7.5, 10, 10],
        [7.5, 7.5, 10, 10],
    ]]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_AveragePoolingBackpropagation_with3Ddim():
    inputTensor = np.array([[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]])  

    errorPatch = np.array([[
        [10, 20],
        [30, 40]
    ]])  

    gridSize = 2
    strideLength = 2
    pool = PoolingLayer(gridSize, strideLength, "ave")

    errorTerm = pool._backpropagation(errorPatch, inputTensor)

    expectedInputErrorTerms = np.array([[
        [2.5, 2.5, 5, 5],
        [2.5, 2.5, 5, 5],
        [7.5, 7.5, 10, 10],
        [7.5, 7.5, 10, 10],
    ]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_GlobalAveragePoolingBackpropagation():
    inputTensor = np.array([[[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]]])
    
    pool = GlobalAveragePooling()
    pool.inputShape = inputTensor.shape

    upstreamGradient = np.ones((1, 1, 1, 1))
    errorTerm = pool._backpropagation(upstreamGradient)

    expectedInputErrorTerms = np.full(inputTensor.shape, fill_value=1 / 16)

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_ActivationLayerBackpropagation():
    from quacknet.core.activations.activationDerivativeFunctions import ReLUDerivative
    inputTensor = np.array([[
        [1, -3, 2, -4],
        [2, -4, -1, 3],
        [4, -1, 3, -2],
        [-3, 2, -4, 1],
    ]])

    errorPatch = np.array([[
        [1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1],
    ]]) 

    errorTerm = ActivationLayer._backpropagation(ReLUDerivative, errorPatch, inputTensor)

    leakyReluDerive = np.where(inputTensor > 0, 1, 0.01)
    expectedInputErrorTerms = errorPatch * leakyReluDerive

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

class Test_DenseLayer:
    def createData(self):
        np.random.seed(55)

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

    def calculateGradients(self, layerNodes, weights, trueValues, activationDerivatives, lossDerivative):
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

    def test_backpropagation(self):
        layerNodes, weights, biases, trueValues, layers = self.createData()
        activationDerivatives = [ReLUDerivative, ReLUDerivative, LinearDerivative]
        lossDerivative = MSEDerivative

        net = Network()
        net.layers = layers
        net.weights = weights
        net.biases = biases

        dense = DenseLayer(net)
        dense.layerNodes = layerNodes
        dense.orignalShape = layerNodes[0].shape

        weightGradients, biasGradients, errorTerms = dense._backpropagation(trueValues)

        expectedWeightGradients, expectedBiasGradients = self.calculateGradients(layerNodes, weights, trueValues, activationDerivatives, lossDerivative)

        _, _, expectedErrorTerms = net._backPropgation(layerNodes, trueValues, True)

        for i in reversed(range(len(net.weights))):
            expectedErrorTerms = net.weights[i] @ expectedErrorTerms
        expectedErrorTerms = expectedErrorTerms.reshape(np.array(layerNodes[0]).shape)

        assert np.allclose(weightGradients[0].shape, expectedWeightGradients[1].shape)
        assert np.allclose(weightGradients[1].shape, expectedWeightGradients[0].shape)
        assert np.allclose(biasGradients[0].shape, expectedBiasGradients[1].shape)
        assert np.allclose(biasGradients[1].shape, expectedBiasGradients[0].shape)

        assert np.allclose(weightGradients[0], expectedWeightGradients[1])
        assert np.allclose(weightGradients[1], expectedWeightGradients[0])
        assert np.allclose(biasGradients[0], expectedBiasGradients[1])
        assert np.allclose(biasGradients[1], expectedBiasGradients[0])

        assert np.allclose(expectedErrorTerms[0].shape, errorTerms[0].shape)
        assert np.allclose(expectedErrorTerms[1].shape, errorTerms[1].shape)

        assert np.allclose(expectedErrorTerms[0], errorTerms[0])
        assert np.allclose(expectedErrorTerms[1], errorTerms[1])
