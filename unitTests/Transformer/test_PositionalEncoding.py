from quacknet import PositionalEncoding
import numpy as np

def test_positional_encoding_shape():
    max_dim = 10
    embed_size = 6
    pos_enc = PositionalEncoding(max_dim, embed_size)
    encoding = pos_enc.encoding

    assert encoding.shape == (max_dim, embed_size)

def test_positional_encoding_values():
    max_dim = 3
    embed_size = 4
    pos_enc = PositionalEncoding(max_dim, embed_size)

    position = np.arange(max_dim)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(100000) / embed_size))

    expected = np.zeros((max_dim, embed_size))
    expected[:, 0::2] = np.sin(position * div_term)
    expected[:, 1::2] = np.cos(position * div_term)

    assert np.allclose(pos_enc.encoding, expected, atol=1e-6)

def test_forward_propagation():
    max_dim = 5
    embed_size = 4
    pos_enc = PositionalEncoding(max_dim, embed_size)

    input_data = np.zeros((1, 3, embed_size))
    output = pos_enc.forwardPropagation(input_data)

    expected = pos_enc.encoding[:3] 
    expected = expected[np.newaxis, :, :]  

    assert output.shape == (1, 3, embed_size) 
    assert np.allclose(output, expected, atol=1e-6) 
