from quacknet import RMSProp
import numpy as np

def test_rmsprop_single_step():
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.1], [0.1, 0.1]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    decay = 0.9
    epsilon = 1e-8

    cache = (1 - decay) * (grad ** 2)
    expectedParam = param - alpha * grad / (np.sqrt(cache) + epsilon)

    rmsprop = RMSProp(None, None, decay=decay, epsilon=epsilon)
    rmsprop.cache = {}
    updatedParam = rmsprop._RMSPropUpdate(Parameters.copy(), Gradients.copy(), alpha)

    np.testing.assert_allclose(updatedParam["W"], expectedParam, rtol=1e-6, atol=1e-8)

def forward(batchData):
    return np.sum(batchData, axis=1)

def backward(output, batchLabels):
    Parameters = {
        "W": np.array([1.0, 1.0]),
        "b": 1.0
    }
    grad_W = np.array([np.mean(output - batchLabels), np.mean(output - batchLabels)])
    Gradients = {
        "W": grad_W,
        "b": 1.0
    }
    return Parameters, Gradients

def fake_rmsprop(params, grads, alpha):
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - alpha * grads[key]
    return updated_params

def test_rmsprop_optimiser_batches():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]),
              np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    labels = [np.array(3.0), np.array(7.0), np.array(11.0), np.array(15.0)]
    batchSize = 2
    alpha = 0.1
    decay = 0.9
    epsilon = 1e-8

    rmsprop = RMSProp(forward, backward, decay=decay, epsilon=epsilon)
    rmsprop._RMSPropUpdate = lambda p, g, a: fake_rmsprop(p, g, a)

    outputs, updated_params = rmsprop._RMSPropOptimiserWithBatches(inputs, labels, batchSize, alpha)

    assert len(outputs) == 2
    expected_outputs = [
        forward(np.array(inputs[0:2])),
        forward(np.array(inputs[2:4]))
    ]
    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)
    for key in updated_params:
        if isinstance(updated_params[key], np.ndarray):
            assert updated_params[key].shape == np.array([1.0, 1.0]).shape

def test_rmsprop_numerical():
    param = np.array([[0.5, -0.5], [1.0, -1.0]])
    grad = np.array([[0.1, -0.2], [0.05, -0.05]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    decay = 0.9
    epsilon = 1e-8

    cache = (1 - decay) * (grad ** 2)
    expected = param - alpha * grad / (np.sqrt(cache) + epsilon)

    rmsprop = RMSProp(None, None, decay=decay, epsilon=epsilon)
    rmsprop.cache = {}
    updated = rmsprop._RMSPropUpdate(Parameters.copy(), Gradients.copy(), alpha)

    np.testing.assert_allclose(updated["W"], expected, rtol=1e-6, atol=1e-8)
