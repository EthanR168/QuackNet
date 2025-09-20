from quacknet import Lion
import numpy as np

def test_lion_single_step():
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.1], [0.1, 0.1]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    beta1 = 0.9

    m_t = (1 - beta1) * grad
    expectedParam = param - alpha * np.sign(m_t)

    lion = Lion(None, None, beta1=beta1)
    lion.momentum = {}
    updatedParam = lion._lionUpdate(Parameters.copy(), Gradients.copy(), alpha)

    np.testing.assert_allclose(updatedParam["W"], expectedParam, rtol=1e-6, atol=1e-8)

def forward(batchData):
    return np.sum(batchData, axis=1)

def backward(output, batchLabels):
    Parameters = {"W": np.array([1.0, 1.0]), "b": 1.0}
    grad_W = np.array([np.mean(output - batchLabels), np.mean(output - batchLabels)])
    Gradients = {"W": grad_W, "b": 1.0}
    return Parameters, Gradients

def fake_lion(params, grads, alpha, beta1=0.9):
    updated_params = {}
    for key in grads:
        if isinstance(grads[key], list):
            updated_params[key] = []
            for i, g in enumerate(grads[key]):
                m = (1 - beta1) * g
                updated_params[key].append(params[key][i] - alpha * np.sign(m))
        else:
            m = (1 - beta1) * grads[key]
            updated_params[key] = params[key] - alpha * np.sign(m)
    return updated_params

def test_lion_optimiser_batches():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    labels = [np.array(3.0), np.array(7.0), np.array(11.0), np.array(15.0)]
    batchSize = 2
    alpha = 0.1

    lion = Lion(forward, backward, beta1=0.9)
    lion._lionUpdate = lambda p, g, a: fake_lion(p, g, a)

    outputs, updated_params = lion._optimiserWithBatches(inputs, labels, batchSize, alpha)

    assert len(outputs) == 2

    expected_outputs = [
        forward(np.array(inputs[0:2])),
        forward(np.array(inputs[2:4]))
    ]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)

def test_lion_numerical():
    param = np.array([[0.5, -0.5], [1.0, -1.0]])
    grad = np.array([[0.1, -0.2], [0.05, -0.05]])
    Parameters = {"W": param.copy()}
    Gradients = {"W": grad.copy()}

    alpha = 0.01
    beta1 = 0.8

    m_t = (1 - beta1) * grad
    expected = param - alpha * np.sign(m_t)

    lion = Lion(None, None, beta1=beta1)
    lion.momentum = {}
    updated = lion._lionUpdate(Parameters.copy(), Gradients.copy(), alpha)

    np.testing.assert_allclose(updated["W"], expected, rtol=1e-6, atol=1e-8)

def test_lion_optimiser_batches_jagged():
    inputs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    labels = [np.array(3.0), np.array(7.0)]
    batchSize = 2
    alpha = 0.1

    def backward_jagged(output, labels):
        Parameters = {"W": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        Gradients = {"W": [np.array([0.1, 0.1]), np.array([0.2, 0.2])]}
        return Parameters, Gradients

    lion = Lion(forward, backward_jagged, beta1=0.9)
    lion._lionUpdate = lambda p, g, a: fake_lion(p, g, a)

    outputs, updated_params = lion._optimiserWithBatches(inputs, labels, batchSize, alpha)

    expected_outputs = [forward(np.array(inputs)), forward(np.array(inputs))[:0]]

    for computed, expected in zip(outputs, expected_outputs):
        np.testing.assert_array_almost_equal(computed, expected)
