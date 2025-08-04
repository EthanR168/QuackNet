from quacknet import FeedForwardNetwork
import numpy as np

def test_forward_propagation():
    input_dim = 2
    hidden_dim = 3

    W1 = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)
    b1 = np.array([[1, 1, 1]], dtype=float)

    W2 = np.array([[1, 0],
                   [0, 1],
                   [1, -1]], dtype=float)
    b2 = np.array([[0, 0]], dtype=float)

    input_tokens = np.array([[[1.0, 2.0], [3.0, 4.0]]])  

    ffn = FeedForwardNetwork(input_dim, hidden_dim, W1, b1, W2, b2)
    ffn.W1 = W1
    ffn.b1 = b1
    ffn.W2 = W2
    ffn.b2 = b2

    output = ffn.forwardPropagation(input_tokens)

    def relu(x): return np.maximum(0, x)

    layer1 = np.matmul(input_tokens, W1) + b1 
    activated = relu(layer1)
    expected_output = np.matmul(activated, W2) + b2  

    assert np.allclose(output, expected_output) 

def test_backward_propagation_shapes():
    input_dim = 4
    hidden_dim = 5
    seq = 3

    input_tokens = np.random.randn(1, seq, input_dim)       
    output_gradient = np.random.randn(1, seq, input_dim)

    ffn = FeedForwardNetwork(input_dim, hidden_dim, None, None, None, None)
    output = ffn.forwardPropagation(input_tokens)
    d_input, dW1, db1, dW2, db2 = ffn.backwardPropagation(output_gradient)

    assert d_input.shape == input_tokens.shape 
    assert dW1.shape == (1, input_dim, hidden_dim)  
    assert db1.shape == (1, 1, hidden_dim)           
    assert dW2.shape == (1, hidden_dim, input_dim)
    assert db2.shape == (1, 1, input_dim)