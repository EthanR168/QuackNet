from quacknet import NormLayer
import numpy as np

def test_forward_output_shape():
    x = np.random.rand(2, 3, 4)
    norm = NormLayer(features=4)
    out = norm.forwardPropagation(x)
    assert out.shape == x.shape

def test_forward_output_normalisation():
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])  
    norm = NormLayer(features=4)
    out = norm.forwardPropagation(x)

    expected = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + norm.epsilon)
    assert np.allclose(out, expected, atol=1e-6)

def test_backward_output_shapes():
    x = np.random.rand(2, 3, 4)
    norm = NormLayer(features=4)
    out = norm.forwardPropagation(x)

    dout = np.random.rand(2, 3, 4)
    dx, dgamma, dbeta = norm.backwardPropagation(dout)

    assert dx.shape == x.shape
    assert dgamma.shape == norm.gamma.shape
    assert dbeta.shape == norm.beta.shape 

def test_backward_zero_gradient_on_constant_input():
    x = np.ones((2, 3, 4))
    dout = np.ones((2, 3, 4))
    norm = NormLayer(features=4)
    norm.forwardPropagation(x)
    dx, dgamma, dbeta = norm.backwardPropagation(dout)

    assert np.allclose(dx, 0, atol=1e-6) 
    assert np.allclose(dgamma, 0, atol=1e-6) 
    assert np.allclose(dbeta, 6 * np.ones((1, 1, 4)))