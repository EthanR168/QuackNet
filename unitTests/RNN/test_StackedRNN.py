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
    batch_size = 4
    sequence_length = 6
    input_size = 5
    output_size = 3
    hidden_sizes = [10, 8]

    rnn = StackedRNN(
        hiddenStateActivationFunction="tanh",
        outputLayerActivationFunction="sigmoid",
        lossFunction="mse",
        numberOfHiddenStates=len(hidden_sizes),
        hiddenSizes=hidden_sizes,
        useBatches=True,
        batchSize=batch_size,
    )
    rnn.initialiseWeights(inputSize=input_size, outputSize=output_size)

    input_data = np.random.randn(batch_size, sequence_length, input_size)
    output = rnn.forwardSequence(input_data)

    assert isinstance(output, np.ndarray)
    assert output.shape == (12, 1)
