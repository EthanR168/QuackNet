from quacknet import SingularRNN
import numpy as np

def test_initialise_weights_shapes():
    rnn = SingularRNN("tanh", "linear", "mse")
    rnn.initialiseWeights(inputSize=5, hiddenSize=4, outputSize=3)

    assert rnn.inputWeight.shape == (4, 5)
    assert rnn.hiddenWeight.shape == (4, 4)
    assert rnn.outputWeight.shape == (3, 4)
    assert rnn.bias.shape == (4, 1)
    assert rnn.outputBias.shape == (3, 1)
    assert rnn.hiddenState.shape == (4, 1)

def test_calculate_hidden_layer_output_shape():
    rnn = SingularRNN("relu", "linear", "mse")
    rnn.initialiseWeights(inputSize=4, hiddenSize=3, outputSize=2)

    x = np.random.randn(4, 1)
    prev_h = np.random.randn(3, 1)

    pre_act, new_h = rnn._calculateHiddenLayer(
        x, prev_h, rnn.inputWeight, rnn.hiddenWeight, rnn.bias, rnn.hiddenStateActivationFunction
    )

    assert pre_act.shape == (1, 3, 1)
    assert new_h.shape == (1, 3, 1)

def test_calculate_output_layer_output_shape():
    rnn = SingularRNN("relu", "linear", "mse")
    rnn.initialiseWeights(inputSize=4, hiddenSize=3, outputSize=2)

    hidden_state = np.random.randn(3, 1)
    pre_act, out = rnn._calculateOutputLayer(
        hidden_state, rnn.outputWeight, rnn.outputBias, rnn.outputLayerActivationFunction
    )

    assert pre_act.shape == (1, 2, 1)
    assert out.shape == (1, 2, 1)

def test_one_step_shapes():
    rnn = SingularRNN("tanh", "sigmoid", "mse")
    rnn.initialiseWeights(inputSize=5, hiddenSize=4, outputSize=3)

    x = np.random.randn(5, 1)
    pre_act, out_pre, out = rnn._oneStep(x)

    assert pre_act.shape == (1, 4, 1)
    assert out_pre.shape == (1, 3, 1)
    assert out.shape == (1, 3, 1)

def test_forward_sequence_output_shape():
    input_size = 5
    hidden_size = 4
    output_size = 3
    batch_size = 2
    sequence_length = 6

    input_data = np.random.randn(batch_size, sequence_length, input_size)

    rnn = SingularRNN("tanh", "linear", "mse")
    rnn.initialiseWeights(input_size, hidden_size, output_size)

    output = rnn.forwardSequence(input_data)

    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 3, 1)
    assert np.all(np.isfinite(output))