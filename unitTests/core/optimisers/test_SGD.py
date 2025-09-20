import numpy as np
from quacknet import SGD

def forward(x):
    return np.sum(x, axis=1)

def backward(output, labels):
    Parameters = {"W": np.array([1.0, 1.0]), "b": 1.0}
    grad_W = np.array([np.mean(output - labels), np.mean(output - labels)])
    Gradients = {"W": grad_W, "b": 0.5}
    return Parameters, Gradients

def test_sgd_single_step():
    inputs = [np.array([1.0, 2.0])]
    labels = [np.array(3.0)]
    sgd = SGD(forward, backward)

    outputs, updated_params = sgd.optimiser(inputs, labels, None, None, learningRate=0.1)

    expected_output = forward(np.array([inputs[0]]))
    np.testing.assert_allclose(outputs[0], expected_output)

    expected_W = np.array([1.0, 1.0]) - 0.1 * np.array([0.0, 0.0])
    np.testing.assert_allclose(updated_params["W"], expected_W)

def test_sgd_multiple_steps():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    labels = [np.array(3.0), np.array(7.0)]
    sgd = SGD(forward, backward)

    outputs, updated_params = sgd.optimiser(inputs, labels, None, None, learningRate=0.1)

    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[0], forward(np.array([inputs[0]])))
    np.testing.assert_array_equal(outputs[1], forward(np.array([inputs[1]])))

    assert "W" in updated_params
    assert "b" in updated_params

def test_sgd_jagged_params():
    def backward_jagged(output, labels):
        Parameters = {"W": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        Gradients = {"W": [np.array([0.1, 0.1]), np.array([0.2, 0.2])]}
        return Parameters, Gradients

    sgd = SGD(forward, backward_jagged)

    inputs = [np.array([1.0, 2.0])]
    labels = [np.array(3.0)]

    _, updated_params = sgd.optimiser(inputs, labels, None, None, learningRate=0.1)

    expected_W = [
        np.array([1.0, 2.0]) - 0.1 * np.array([0.1, 0.1]),
        np.array([3.0, 4.0]) - 0.1 * np.array([0.2, 0.2])
    ]
    for w, exp in zip(updated_params["W"], expected_W):
        np.testing.assert_allclose(w, exp)
