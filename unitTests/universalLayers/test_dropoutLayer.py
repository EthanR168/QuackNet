import numpy as np
from quacknet import Dropout  

def test_forward_output_shape():
    x = np.random.rand(5, 4, 3)
    dropout = Dropout(dropProbability=0.5)
    out = dropout.forward(x, training=True)

    assert out.shape == x.shape

def test_forward_mask_applied():
    x = np.ones((10, 10))
    dropout = Dropout(dropProbability=0.5)
    out = dropout.forward(x, training=True)
    
    assert np.any(out == 0)
    assert np.all(out <= x / (1 - dropout.dropProbability))

def test_forward_no_dropout_in_inference():
    x = np.random.rand(3, 3)
    dropout = Dropout(dropProbability=0.5)
    out = dropout.forward(x, training=False)

    assert np.allclose(out, x)

def test_backward_gradient_masking():
    x = np.random.rand(4, 4)
    grad_output = np.ones_like(x)
    dropout = Dropout(dropProbability=0.5)
    dropout.forward(x, training=True)
    grad_input = dropout._backpropagation(grad_output)
    
    zero_mask_indices = dropout.mask == 0
    one_mask_indices = dropout.mask == 1

    assert np.all(grad_input[zero_mask_indices] == 0)
    assert np.allclose(grad_input[one_mask_indices], 1 / (1 - dropout.dropProbability))

def test_forward_dropProb_Zero():
    x = np.ones((2, 2))
    dropout = Dropout(dropProbability=0)
    out = dropout.forward(x, training=True)

    assert np.allclose(out, x)

def test_forward_dropProb_One():
    x = np.ones((2, 2))
    dropout = Dropout(dropProbability=1.0)
    out = dropout.forward(x, training=True)

    assert np.all(out == 0)
