from quacknet.RNN.rnnManager import RNN
import numpy as np

def test_Singular_BPTT_No_OutputLayer():
    inputWeights = np.array([[0.1, 0.2], [0.3, 0.4]])
    hiddenWeights = np.array([[0.5, 0.0], [0.0, 0.5]])
    biases = np.array([0.0, 0.0])

    model = RNN(1, activationFunction="tanh", lossFunc="MSE")

    model.inputWeights = inputWeights
    model.hiddenWeights = hiddenWeights
    model.biases = biases
    
    x0 = np.array([1.0, 2.0])
    inputs = [x0]

    t0 = np.array([0.5, -0.5])
    targets = [t0]

    z0 = inputWeights @ x0 + biases
    h0 = np.tanh(z0)
    hiddenStates = [h0]
    preActivationValues = [z0]

    lossDeriv = model.lossDerivative(h0, t0, len(t0))
    activationDeriv = model.activationDerivative(z0)
    dz = lossDeriv * activationDeriv

    expectedInputGrad = np.outer(dz, x0)
    expectedBiasGrad = dz
    expectedHiddenGrad = np.zeros_like(hiddenWeights)

    inputGrad, hiddenGrad, biasGrad = model._Singular_BPTTNoOutputLayer(inputs, hiddenStates, preActivationValues, targets)

    assert inputGrad.shape == inputWeights.shape
    assert hiddenGrad.shape == hiddenWeights.shape
    assert biasGrad.shape == biases.shape

    assert np.allclose(inputGrad, expectedInputGrad)
    assert np.allclose(biasGrad, expectedBiasGrad)
    assert np.allclose(hiddenGrad, expectedHiddenGrad)

def test_Singular_BPTT_With_OutputLayer():
    inputWeights = np.array([[0.1, 0.2], [0.3, 0.4]])
    hiddenWeights = np.array([[0.5, 0.0], [0.0, 0.5]])
    outputWeights = np.array([[0.6, 0.7], [0.8, 0.9]])
    biases = np.array([0.0, 0.0])
    outputBiases = np.array([0.0, 0.0])

    model = RNN(1, activationFunction="tanh", lossFunc="MSE")

    model.inputWeights = inputWeights
    model.hiddenWeights = hiddenWeights
    model.outputWeights = outputWeights
    model.biases = biases
    model.outputBiases = outputBiases
    
    x0 = np.array([1.0, 2.0])
    inputs = [x0]

    t0 = np.array([0.5, -0.5])
    targets = [t0]

    z0 = inputWeights @ x0 + biases
    h0 = np.tanh(z0)
    hiddenStates = [h0]
    preActivationValues = [z0]

    output0 = outputWeights @ h0 + outputBiases
    outputs = [output0]

    outputLoss = model.lossDerivative(output0, t0, len(t0))
    expectedOutputWeightGrad = np.outer(outputLoss, h0)
    expectedOutputBiasGrad = outputLoss

    delta = (outputLoss @ outputWeights) * model.activationDerivative(z0)
    expectedInputGrad = np.outer(delta, x0)
    expectedHiddenGrad = np.zeros_like(hiddenWeights)
    expectedBiasGrad = delta

    inputGrad, hiddenGrad, biasGrad, outputGrad, outputBiasGrad = model._Singular_BPTTWithOutputLayer(inputs, hiddenStates, preActivationValues, targets, outputs)

    assert inputGrad.shape == inputWeights.shape
    assert hiddenGrad.shape == hiddenWeights.shape
    assert biasGrad.shape == biases.shape
    assert outputGrad.shape == outputWeights.shape
    assert outputBiasGrad.shape == outputBiases.shape

    assert np.allclose(inputGrad, expectedInputGrad)
    assert np.allclose(biasGrad, expectedBiasGrad)
    assert np.allclose(hiddenGrad, expectedHiddenGrad)
    assert np.allclose(outputGrad, expectedOutputWeightGrad)
    assert np.allclose(outputBiasGrad, expectedOutputBiasGrad)

def test_Stacked_BPTT_No_OutputLayer():
    inputWeights = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5, 0.6], [0.7, 0.8]]),
    ]
    hiddenWeights = [
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    ]
    biases = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0])
    ]

    model = RNN(1, activationFunction="tanh", lossFunc="MSE")

    model.inputWeights = inputWeights
    model.hiddenWeights = hiddenWeights
    model.biases = biases
    model.layers = [2, 2]
    
    x0 = np.array([1.0, 2.0])
    inputs = [x0]

    t0 = np.array([0.5, -0.5])
    targets = [t0]

    z0 = inputWeights[0] @ x0 + biases[0]
    h0 = np.tanh(z0)

    z1 = inputWeights[1] @ h0 + biases[1]
    h1 = np.tanh(z1)   

    hiddenStates = [[h0], [h1]]
    preActivationValues = [[z0], [z1]]

    lossDeriv = model.lossDerivative(h1, t0, len(t0))
    dz1 = lossDeriv * model.activationDerivative(z1)
    dz0 = (dz1 @ inputWeights[1]) * model.activationDerivative(z0)

    expectedInputGrad0 = np.outer(dz0, x0)
    expectedInputGrad1 = np.outer(dz1, h0)

    expectedBiasGrad0 = dz0
    expectedBiasGrad1 = dz1

    expectedHiddenGrad0 = np.zeros_like(hiddenWeights[0])
    expectedHiddenGrad1 = np.zeros_like(hiddenWeights[1])

    inputGrad, hiddenGrad, biasGrad = model._Stacked_BPTTNoOutputLayer(inputs, hiddenStates, preActivationValues, targets)

    assert inputGrad[0].shape == inputWeights[0].shape
    assert inputGrad[1].shape == inputWeights[1].shape
    assert hiddenGrad[0].shape == hiddenWeights[0].shape
    assert hiddenGrad[1].shape == hiddenWeights[1].shape
    assert biasGrad[0].shape == biases[0].shape
    assert biasGrad[1].shape == biases[1].shape

    assert np.allclose(inputGrad[0], expectedInputGrad0)
    assert np.allclose(inputGrad[1], expectedInputGrad1)
    assert np.allclose(biasGrad[0], expectedBiasGrad0)
    assert np.allclose(biasGrad[1], expectedBiasGrad1)
    assert np.allclose(hiddenGrad[0], expectedHiddenGrad0)
    assert np.allclose(hiddenGrad[1], expectedHiddenGrad1)

def test_Stacked_BPTT_With_OutputLayer():
    inputWeights = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5, 0.6], [0.7, 0.8]]),
    ]
    hiddenWeights = [
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    ]
    biases = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0])
    ]
    outputWeights = np.array([[0.6, 0.7], [0.8, 0.9]])
    outputBiases = np.array([0.0, 0.0])

    model = RNN(1, activationFunction="tanh", lossFunc="MSE")

    model.inputWeights = inputWeights
    model.hiddenWeights = hiddenWeights
    model.biases = biases
    model.layers = [2, 2]
    model.outputWeights = outputWeights
    model.outputBiases = outputBiases
    
    x0 = np.array([1.0, 2.0])
    inputs = [x0]

    t0 = np.array([0.5, -0.5])
    targets = [t0]

    z0 = inputWeights[0] @ x0 + biases[0]
    h0 = np.tanh(z0)

    z1 = inputWeights[1] @ h0 + biases[1]
    h1 = np.tanh(z1)   

    o0 = outputWeights @ h1 + outputBiases
    outputs = [o0]

    hiddenStates = [[h0], [h1]]
    preActivationValues = [[z0], [z1]]

    outputLoss = model.lossDerivative(o0, t0, len(t0))
    expectedOutputWeightGrad = np.outer(outputLoss, h1)
    expectedOutputBiasGrad = outputLoss

    dz1 = (outputLoss @ outputWeights) * model.activationDerivative(z1)
    dz0 = (dz1 @ inputWeights[1]) * model.activationDerivative(z0)

    expectedInputGrad0 = np.outer(dz0, x0)
    expectedInputGrad1 = np.outer(dz1, h0)

    expectedBiasGrad0 = dz0
    expectedBiasGrad1 = dz1

    expectedHiddenGrad0 = np.zeros_like(hiddenWeights[0])
    expectedHiddenGrad1 = np.zeros_like(hiddenWeights[1])

    inputGrad, hiddenGrad, biasGrad, outputGrads, outputBiasGrad = model._Stacked_BPTTWithOutputLayer(inputs, hiddenStates, preActivationValues, targets, outputs)

    assert inputGrad[0].shape == inputWeights[0].shape
    assert inputGrad[1].shape == inputWeights[1].shape
    assert hiddenGrad[0].shape == hiddenWeights[0].shape
    assert hiddenGrad[1].shape == hiddenWeights[1].shape
    assert biasGrad[0].shape == biases[0].shape
    assert biasGrad[1].shape == biases[1].shape
    assert outputGrads.shape == outputWeights.shape
    assert outputBiasGrad.shape == outputBiases.shape

    assert np.allclose(inputGrad[0], expectedInputGrad0)
    assert np.allclose(inputGrad[1], expectedInputGrad1)
    assert np.allclose(biasGrad[0], expectedBiasGrad0)
    assert np.allclose(biasGrad[1], expectedBiasGrad1)
    assert np.allclose(hiddenGrad[0], expectedHiddenGrad0)
    assert np.allclose(hiddenGrad[1], expectedHiddenGrad1)
    assert np.allclose(outputGrads, expectedOutputWeightGrad)
    assert np.allclose(outputBiasGrad, expectedOutputBiasGrad)

