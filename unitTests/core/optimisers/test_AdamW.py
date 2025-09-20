from quacknet import AdamW
import numpy as np

def test_adamw_single_step():
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.1], [0.1, 0.1]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.1

    t = 1
    m = (1 - beta1) * grad
    v = (1 - beta2) * (grad ** 2)
    mHat = m / (1 - beta1 ** t)
    vHat = v / (1 - beta2 ** t)

    expectedParam = param - alpha * mHat / (np.sqrt(vHat) + epsilon)
    expectedParam -= alpha * weight_decay * expectedParam

    adamw = AdamW(None, None, weightDecay=weight_decay)
    adamw.t = 0
    updatedParam = adamw._AdamW(Parameters.copy(), Gradients.copy(), alpha, beta1, beta2, epsilon)

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

def fake_adamw(params, grads, alpha, beta1, beta2, epsilon, weight_decay=0.01):
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - alpha * grads[key] - alpha * weight_decay * params[key]
    return updated_params

def test_adamw_optimiser_batches():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    labels = [np.array(3.0), np.array(7.0), np.array(11.0), np.array(15.0)]
    batchSize = 2
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.01

    adamw = AdamW(forward, backward, weightDecay=weight_decay)
    adamw._AdamW = lambda p, g, a, b1, b2, eps: fake_adamw(p, g, a, b1, b2, eps, weight_decay)  # override

    outputs, updated_params = adamw._AdamWOptimiserWithBatches(inputs, labels, batchSize, alpha, beta1, beta2, epsilon)

    assert len(outputs) == 2

    expected_outputs = [
        forward(np.array(inputs[0:2])),
        forward(np.array(inputs[2:4]))
    ]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)

    for key in updated_params:
        if isinstance(updated_params[key], np.ndarray):
            assert np.all(updated_params[key] <= 1.0)

def test_adamw_numerical():
    param = np.array([[0.5, -0.5], [1.0, -1.0]])
    grad = np.array([[0.1, -0.2], [0.05, -0.05]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.05

    t = 1
    m = (1 - beta1) * grad  
    v = (1 - beta2) * (grad ** 2)  
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    expected = param - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    expected -= alpha * weight_decay * expected 

    adamw = AdamW(None, None, weightDecay=weight_decay)
    adamw.t = 0
    updated = adamw._AdamW(Parameters.copy(), Gradients.copy(), alpha, beta1, beta2, epsilon)

    np.testing.assert_allclose(updated["W"], expected, rtol=1e-6, atol=1e-8)