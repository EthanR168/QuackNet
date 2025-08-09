import numpy as np
from quacknet import SGD  

def forward(x):
    return np.sum(x)

def backward(layerNodes, label):
    Parameters = {
        "W": np.array([1.0, 1.0]),
        "b": 1.0
    }
    grad_val = layerNodes - label
    Gradients = {
        "W": np.array([grad_val, grad_val]),
        "b": 1.0
    }
    return Parameters, Gradients

def test_SGD_withoutBatches():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    labels = [3.0, 7.0]
    learningRate = 0.1

    sgd = SGD(forward, backward)
    sgd.giveInputsToBackprop = False

    allNodes, updated_params = sgd.optimiser(inputs, labels, useBatches=False, batchSize=None, learningRate=learningRate)

    assert len(allNodes) == 2
    assert np.allclose(allNodes[0], forward(inputs[0]))
    assert np.allclose(allNodes[1], forward(inputs[1]))

    last_grad_val = allNodes[-1] - labels[-1]
    expected_W = np.array([1.0, 1.0]) - learningRate * np.array([last_grad_val, last_grad_val])
    expected_b = 1.0 - learningRate * 1.0

    assert np.allclose(updated_params["W"], expected_W)
    assert np.allclose(updated_params["b"], expected_b)

def test_sgd_with_batches():
    inputs = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
        np.array([7.0, 8.0])
    ]
    labels = [3.0, 7.0, 11.0, 15.0]
    batchSize = 2
    learningRate = 0.1

    sgd = SGD(forward, backward)
    sgd.giveInputsToBackprop = False

    allNodes, updated_params = sgd.optimiser(inputs, labels, useBatches=True, batchSize=batchSize, learningRate=learningRate)

    assert len(allNodes) == len(inputs)

    for i in range(len(inputs)):
        assert np.allclose(allNodes[i], forward(inputs[i]))

    start_idx = len(inputs) - batchSize  
    grad_sum_W = np.zeros(2)
    grad_sum_b = 0.0

    for i in range(start_idx, len(inputs)):
        forward_out = forward(inputs[i])
        grad_val = forward_out - labels[i]  
        grad_sum_W += np.array([grad_val, grad_val])
        grad_sum_b += 1.0

    grad_avg_W = grad_sum_W / batchSize
    grad_avg_b = grad_sum_b / batchSize

    expected_W = np.array([1.0, 1.0]) - learningRate * grad_avg_W
    expected_b = 1.0 - learningRate * grad_avg_b

    assert np.allclose(updated_params["W"], expected_W, rtol=1e-5, atol=1e-8)
    assert np.allclose(updated_params["b"], expected_b, rtol=1e-5, atol=1e-8)