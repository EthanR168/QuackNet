from quacknet.CNN.cnnOptimiser import CNNoptimiser
from quacknet.main import Network
import numpy as np

from quacknet.CNN.manager import CNNModel

def test_adamsWeightBiasUpdateWithoutBatches():
    weights = [[np.array([1.0])]]
    biases = [[np.array([0.5])]]

    inputData = [1.0]
    labels = [0.0]

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    timestamp = 1

    gradWeight = np.array([0.1])
    gradBias = np.array([0.01])

    mWeight = beta1 * 0 + (1 - beta1) * gradWeight
    vWeight = beta2 * 0 + (1 - beta2) * (gradWeight ** 2)

    mWeightHat = mWeight / (1 - beta1 ** timestamp)
    vWeightHat = vWeight / (1 - beta2 ** timestamp)

    expectedWeight = weights[0][0] - alpha * mWeightHat / (np.sqrt(vWeightHat) + epsilon)

    mBias = beta1 * 0 + (1 - beta1) * gradBias
    vBias = beta2 * 0 + (1 - beta2) * (gradBias ** 2)

    mBiastHat = mBias / (1 - beta1 ** timestamp)
    vBiasHat = vBias / (1 - beta2 ** timestamp)

    expectedBias = biases[0][0] - alpha * mBiastHat / (np.sqrt(vBiasHat) + epsilon)

    cnn = CNNoptimiser()

    cnn._backpropagation = lambda nodes, label: ([[gradWeight]], [[gradBias]])
    cnn.forward = lambda input: ("Doesnt matter since backprogation is harcoded")

    _, weights, biases = cnn._AdamsOptimiserWithoutBatches(inputData, labels, weights, biases, alpha, beta1, beta2, epsilon)

    assert np.allclose(weights[0][0], expectedWeight)
    assert np.allclose(biases[0][0], expectedBias)

def test_adamsWeightBiasUpdateWithBatches():
    weights = [[np.array([1.0])]]
    biases = [[np.array([0.5])]]

    inputData = [1.0, 2.0]
    labels = [0.0, 0.0]

    batchSize = 2

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    timestamp = 1

    gradWeightBatch = [np.array([0.1]), np.array([0.2])]
    gradBiasBatch = [np.array([0.01]), np.array([0.02])]

    avgGradWeight = sum(gradWeightBatch) / len(gradWeightBatch)
    avgGradBias = sum(gradBiasBatch) / len(gradBiasBatch)

    mWeight = beta1 * 0 + (1 - beta1) * avgGradWeight
    vWeight = beta2 * 0 + (1 - beta2) * (avgGradWeight ** 2)

    mWeightHat = mWeight / (1 - beta1 ** timestamp)
    vWeightHat = vWeight / (1 - beta2 ** timestamp)

    expectedWeight = weights[0][0] - alpha * mWeightHat / (np.sqrt(vWeightHat) + epsilon)

    mBias = beta1 * 0 + (1 - beta1) * avgGradBias
    vBias = beta2 * 0 + (1 - beta2) * (avgGradBias ** 2)

    mBiastHat = mBias / (1 - beta1 ** timestamp)
    vBiasHat = vBias / (1 - beta2 ** timestamp)

    expectedBias = biases[0][0] - alpha * mBiastHat / (np.sqrt(vBiasHat) + epsilon)

    cnn = CNNoptimiser()

    cnn._backpropagation = lambda nodes, label: ([[gradWeightBatch.pop(0)]], [[gradBiasBatch.pop(0)]])
    cnn.forward = lambda input: ("Doesnt matter since backprogation is harcoded")

    _, weights, biases = cnn._AdamsOptimiserWithBatches(inputData, labels, weights, biases, batchSize, alpha, beta1, beta2, epsilon)

    assert np.allclose(weights[0][0], expectedWeight)
    assert np.allclose(biases[0][0], expectedBias)