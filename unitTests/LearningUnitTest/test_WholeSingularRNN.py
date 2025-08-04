from quacknet import SingularRNN
import numpy as np

def test_Singular_CheckIfModelCanLearnOnSimpleData():
    rnn = SingularRNN(hiddenStateActivationFunction="tanh", outputLayerActivationFunction="sigmoid", lossFunction="mse")

    inputs = [ 
        [np.array([[0]]), np.array([[0]])],
        [np.array([[0]]), np.array([[1]])], 
        [np.array([[1]]), np.array([[0]])],
        [np.array([[1]]), np.array([[1]])], 
    ]

    targets = [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[1]]),
        np.array([[0]]),
    ]

    epochs = 100

    rnn.initialiseWeights(inputSize=1, hiddenSize=64, outputSize=1)

    initialLoss = rnn.train(inputs, targets)

    for i in range(epochs):
        loss = rnn.train(inputs, targets)

    assert loss < initialLoss