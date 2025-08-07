from quacknet import Conv2DLayer, PoolingLayer, GlobalAveragePooling, ActivationLayer
import numpy as np

def test_ConvulutionalDerivative():
    stride = 1

    inputTensor = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    kernals = np.array([[[[1, 0], [0, -1]]]])
    errorPatch = np.array([[[[1, 2], [3, 4]]]])

    Conv = Conv2DLayer(2, 1, 1, stride)

    Conv.kernalSize = 2
    Conv.kernalWeights = kernals
    Conv.stride = stride

    weightGradients, biasGradients, errorTerms = Conv._backpropagation(errorPatch, inputTensor)

    expectedWeightGradients = np.array([[[[37, 47], [67, 77]]]])
    expectedBiasGradients = np.array([10])
    expectedInputErrorTerms = np.array([[[-1, -2, 0], [-3, -3, 2], [0, 3, 4]]])

    assert expectedWeightGradients.shape == weightGradients.shape
    assert expectedBiasGradients.shape == biasGradients.shape

    assert np.allclose(weightGradients, expectedWeightGradients)
    assert np.allclose(biasGradients, expectedBiasGradients)
    assert np.allclose(errorTerms, expectedInputErrorTerms)

def test_MaxPooling():
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

    errorTerm = pool._MaxPoolingDerivative(errorPatch, inputTensor, gridSize, strideLength)

    expectedInputErrorTerms = np.array([[[
        [0, 0, 0, 20],
        [0, 10, 0, 0],
        [30, 0, 0, 0],
        [0, 0, 40, 0],
    ]]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_AveragePooling():
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

    errorTerm = pool._AveragePoolingDerivative(errorPatch, inputTensor, gridSize, strideLength)

    expectedInputErrorTerms = np.array([[[
        [2.5, 2.5, 5, 5],
        [2.5, 2.5, 5, 5],
        [7.5, 7.5, 10, 10],
        [7.5, 7.5, 10, 10],
    ]]])

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_GlobalAveragePooling():
    inputTensor = np.array([[[[
        1, 3, 2, 4],
        [2, 4, 1, 3],
        [4, 1, 3, 2],
        [3, 2, 4, 1]
    ]]]).astype(np.float64)

    gap = GlobalAveragePooling()
    gap.inputShape = inputTensor.shape

    outputGradient = np.array([[[1]]])  # simulate dL/dOutput = 1
    errorTerm = gap._backpropagation(outputGradient)

    expectedInputErrorTerms = np.full(inputTensor.shape, fill_value=1 / 16)

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)

def test_ActivationLayer():
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

    errorTerm = ActivationLayer._backpropagation(None, errorPatch, inputTensor)

    leakyReluDerive = np.where(inputTensor > 0, 1, 0.01)
    expectedInputErrorTerms = errorPatch * leakyReluDerive

    assert errorTerm.shape == expectedInputErrorTerms.shape
    assert np.allclose(errorTerm, expectedInputErrorTerms)