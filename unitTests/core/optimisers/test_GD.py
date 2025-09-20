import numpy as np
from quacknet import GD

def forward(batchData):
    return np.sum(batchData, axis=1)

def backward(output, batchLabels):
    Parameters = {"W": np.array([1.0, 1.0]), "b": 1.0}
    grad_W = np.array([np.mean(output - batchLabels), np.mean(output - batchLabels)])
    Gradients = {"W": grad_W, "b": 1.0}
    return Parameters, Gradients


def test_gd_update_step():
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, -0.2])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}
    lr = 0.05

    expected = param - lr * grad

    gd = GD(forward, backward)
    updated = gd._updateWeightsBiases(Parameters.copy(), Gradients.copy(), lr)

    np.testing.assert_allclose(updated["W"], expected, rtol=1e-6, atol=1e-8)


def test_gd_optimiser_batches():
    inputs = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
        np.array([7.0, 8.0]),
    ]
    labels = [np.array(3.0), np.array(7.0), np.array(11.0), np.array(15.0)]
    batchSize = 2
    lr = 0.1

    gd = GD(forward, backward)

    outputs, updated_params = gd._trainGradientDescent_Batching(inputs, labels, batchSize, lr)

    assert len(outputs) == 2  

    expected_outputs = [
        forward(np.array(inputs[0:2])),
        forward(np.array(inputs[2:4])),
    ]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)


def test_gd_numerical():
    param = np.array([0.5, -0.5])
    grad = np.array([0.1, -0.2])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}
    lr = 0.01

    expected = param - lr * grad

    gd = GD(forward, backward)
    updated = gd._updateWeightsBiases(Parameters.copy(), Gradients.copy(), lr)

    np.testing.assert_allclose(updated["W"], expected, rtol=1e-6, atol=1e-8)


def test_gd_optimiser_batches_jagged():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    labels = [np.array(3.0), np.array(7.0)]
    batchSize = 2
    lr = 0.1

    def backward_jagged(output, labels):
        Parameters = {"W": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        Gradients = {"W": [np.array([0.1, 0.1]), np.array([0.2, 0.2])]}
        return Parameters, Gradients

    gd = GD(forward, backward_jagged)

    outputs, updated_params = gd._trainGradientDescent_Batching(inputs, labels, batchSize, lr)

    expected_outputs = [forward(np.array(inputs))]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)
