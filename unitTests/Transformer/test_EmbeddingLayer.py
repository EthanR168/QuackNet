from quacknet import EmbeddingLayer
import numpy as np

def test_forward_output_values():
    layer = EmbeddingLayer(vocabSize=10, embedDimension=4)
    layer.weights = np.arange(40).reshape(10, 4)  

    input_indices = np.array([1, 3, 7])
    expected_output = layer.weights[input_indices]

    output = layer.forwardPropagation(input_indices)
    assert np.array_equal(output, expected_output) 

def test_forward_output_shape_batch():
    layer = EmbeddingLayer(vocabSize=20, embedDimension=5)
    input_indices = np.array([[1, 2], [3, 4]])
    output = layer.forwardPropagation(input_indices)

    assert output.shape == (2, 2, 5) 

def test_backward_updates():
    layer = EmbeddingLayer(vocabSize=5, embedDimension=3)
    layer.weights = np.ones((5, 3))  

    input_indices = np.array([[0, 1], [2, 0]])
    layer.forwardPropagation(input_indices)

    grad_out = np.array([
        [[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]]
    ])

    grad_weights = layer.backwardPropagation(grad_out)

    expected = np.zeros((5, 3))
    expected[0] += grad_out[0, 0]   
    expected[1] += grad_out[0, 1]   
    expected[2] += grad_out[1, 0]   
    expected[0] += grad_out[1, 1]   

    assert np.array_equal(grad_weights, expected) 
