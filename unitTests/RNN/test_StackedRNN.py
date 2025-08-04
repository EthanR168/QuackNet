from quacknet import StackedRNN
import numpy as np

def test_initialise_weights_shapes():
    rnn = StackedRNN("tanh", "linear", "mse", numberOfHiddenStates=2, hiddenSizes=[3, 2])
    rnn.initialiseWeights(inputSize=4, outputSize=1)

    assert len(rnn.inputWeights) == 2
    assert rnn.inputWeights[0].shape == (3, 4)
    assert rnn.inputWeights[1].shape == (2, 3)
    assert rnn.outputWeight.shape == (1, 2)
    assert rnn.outputBias.shape == (1, 1)

def test_forward_shape():
    rnn = StackedRNN("tanh", "linear", "mse", numberOfHiddenStates=2, hiddenSizes=[3, 2])
    rnn.initialiseWeights(inputSize=4, outputSize=1)

    input_seq = [np.random.randn(4, 1) for _ in range(5)]
    output = rnn.forwardSequence(input_seq)
    assert output.shape == (1, 1)
