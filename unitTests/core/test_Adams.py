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