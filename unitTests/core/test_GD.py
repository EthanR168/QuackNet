from quacknet import GD
import numpy as np

def forward(x):
    return x * 2

def backward(layerNodes, label):
    Parameters = {
        "W": np.array([1.0, 1.0]),
        "b": 1.0
    }
    Gradients = {
        "W": layerNodes - label,
        "b": 1.0
    }

    return Parameters, Gradients

def test_GDOptimiser():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    labels = [np.array([1.5, 3.0]), np.array([6.0, 8.0])]
    gd = GD(forward, backward, giveInputsToBackprop=False)
    
    allNodes, updated_params = gd.optimiser(inputs, labels, useBatches=False, batchSize=None, learningRate=0.1)

    expected_nodes = [forward(x) for x in inputs]
    for computed, expected in zip(allNodes, expected_nodes):
        assert np.allclose(computed, expected)

    grad_W_0 = expected_nodes[0] - labels[0]
    grad_W_1 = expected_nodes[1] - labels[1]
    avg_grad_W = (grad_W_0 + grad_W_1) / 2
    expected_W = np.array([1.0, 1.0]) - 0.1 * avg_grad_W
    expected_b = 1.0 - 0.1 * 1.0  

    assert np.allclose(updated_params["W"], expected_W)
    assert np.allclose(updated_params["b"], expected_b)