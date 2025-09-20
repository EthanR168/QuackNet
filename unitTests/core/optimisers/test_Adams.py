from quacknet import Adam
import numpy as np

def test_adams():
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.1], [0.1, 0.1]])
    Parameters = {
        "W": param.copy()
    }
    Gradients = {
        "W": grad.copy()
    }

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    t = 1
    m = (1 - beta1) * grad
    v = (1 - beta2) * (grad ** 2)
    mHat = m / (1 - beta1 ** t)
    vHat = v / (1 - beta2 ** t)
    expectedParam = param - alpha * mHat / (np.sqrt(vHat) + epsilon)

    adam = Adam(None, None)
    adam.t = 0
    updatedParam = adam._Adams(Parameters.copy(), Gradients.copy(), alpha, beta1, beta2, epsilon)

    assert np.allclose(Parameters["W"], updatedParam["W"])

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

def fake_adams(params, grads, alpha, beta1, beta2, epsilon):
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - alpha * grads[key]
    return updated_params

def test_adamsOptimiser():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    labels = [np.array(3.0), np.array(7.0), np.array(11.0), np.array(15.0)]
    batchSize = 2
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    gd = Adam(forward, backward)
    gd._Adams = fake_adams #overwritting the function because i cant be bothered
    gd.giveInputsToBackprop = False

    outputs, updated_params = gd._AdamsOptimiserWithBatches(inputs, labels, batchSize, alpha, beta1, beta2, epsilon)

    assert len(outputs) == 2

    expected_outputs = [
        forward(np.array(inputs[0:2])),
        forward(np.array(inputs[2:4]))
    ]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)

    batch1_grad = np.mean(expected_outputs[0] - np.array(labels[0:2]))
    batch2_grad = np.mean(expected_outputs[1] - np.array(labels[2:4]))
    expected_grad_W_1 = np.array([batch1_grad, batch1_grad]) / batchSize
    expected_grad_W_2 = np.array([batch2_grad, batch2_grad]) / batchSize

    expected_W = np.array([1.0, 1.0]) - alpha * (np.array([batch2_grad, batch2_grad]) / batchSize)
    expected_b = 1.0 - alpha * (1.0 / batchSize)

    assert np.allclose(updated_params["W"], expected_W)
    assert np.allclose(updated_params["b"], expected_b)