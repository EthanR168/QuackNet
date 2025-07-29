from quacknet.RNN.rnnOptimiser import RNNOptimiser
import numpy as np

def test_adam():
    weights = [np.array([[0.5, -0.2], [0.1, 0.3]])]
    biases = [np.array([0.1, -0.1])]

    weightGrads = [np.array([[0.02, -0.01], [0.03, 0.00]])]
    biasGrads = [np.array([0.005, -0.005])]

    firstMomentW = [np.zeros_like(weights[0])]
    firstMomentB = [np.zeros_like(biases[0])]
    secondMomentW = [np.zeros_like(weights[0])]
    secondMomentB = [np.zeros_like(biases[0])]

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    timestep = 1

    mw = beta1 * firstMomentW[0] + (1 - beta1) * weightGrads[0]
    mb = beta1 * firstMomentB[0] + (1 - beta1) * biasGrads[0]

    vw = beta2 * secondMomentW[0] + (1 - beta2) * (weightGrads[0] ** 2)
    vb = beta2 * secondMomentB[0] + (1 - beta2) * (biasGrads[0] ** 2)

    mwHat = mw / (1 - beta1 ** timestep)
    mbHat = mb / (1 - beta1 ** timestep)

    vwHat = vw / (1 - beta2 ** timestep)
    vbHat = vb / (1 - beta2 ** timestep)

    expectedWeights = weights[0] - alpha * mwHat / (np.sqrt(vwHat) + epsilon)
    expectedBiases = biases[0] - alpha * mbHat / (np.sqrt(vbHat) + epsilon)

    optimiser = RNNOptimiser()
    updatedWeights, updatedBiases, _, _, _, _ = optimiser._Adams(weightGrads, biasGrads, weights, biases, timestep, firstMomentW, firstMomentB, secondMomentW, secondMomentB, alpha, beta1, beta2, epsilon)

    assert np.allclose(updatedWeights[0], expectedWeights)
    assert np.allclose(updatedBiases[0], expectedBiases)

def test_AdamsOptimiser_WithoutBatches():
    optimiser = RNNOptimiser()

    optimiser.RNNForwardPropagation = lambda input: input

    optimiser.RNNBackwardPropagation = lambda layerNodes, label: (
        np.array([[0.1]]), np.array([[0.05]]), np.array([0.01]), np.array([[0.2]]), np.array([0.02])
    )

    inputData = [np.array([[1.0]])]
    labels = [np.array([1])]

    inputWeights = [np.array([[0.5]])]
    hiddenWeights = [np.array([[0.25]])]
    biases = [np.array([0.1])]
    outputWeights = [np.array([[0.75]])]
    outputBiases = [np.array([0.2])]

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    allNodes, newInputWeights, newHiddenWeights, newBiases, newOutputWeights, newOutputBias = optimiser._AdamsOptimiserWithoutBatches(inputData, labels, inputWeights, hiddenWeights, biases, outputWeights, outputBiases, alpha, beta1, beta2, epsilon) 

    expectedInputWeight = inputWeights[0] - alpha * 0.1 / (np.sqrt(0.1 ** 2) + epsilon)
    expectedHiddenWeight = hiddenWeights[0] - alpha * 0.05 / (np.sqrt(0.05**2) + epsilon)
    expectedOutputWeight = outputWeights[0] - alpha * 0.2 / (np.sqrt(0.2 ** 2) + epsilon)
    expectedBias = biases[0] - alpha * 0.01 / (np.sqrt(0.01 ** 2) + epsilon)
    expectedOutputBias = outputBiases[0] - alpha * 0.02 / (np.sqrt(0.02**2) + epsilon)

    assert np.allclose(newInputWeights[0], expectedInputWeight)
    assert np.allclose(newHiddenWeights[0], expectedHiddenWeight)
    assert np.allclose(newOutputWeights[0], expectedOutputWeight)
    assert np.allclose(newBiases[0], expectedBias)
    assert np.allclose(newOutputBias[0], expectedOutputBias)