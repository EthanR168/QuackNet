from quacknet import StackedRNN
import numpy as np

def test_Stacked_CheckIfModelCanLearnOnSimpleData():
    rnn = StackedRNN(hiddenStateActivationFunction="tanh", outputLayerActivationFunction="sigmoid", lossFunction="cross", numberOfHiddenStates=2, hiddenSizes=[4, 4])

    inputs = [ # Simple XOR
        [np.array([[0]]), np.array([[0]])], # 0 XOR 0 = 0
        [np.array([[0]]), np.array([[1]])], # 0 XOR 1 = 1
        [np.array([[1]]), np.array([[0]])], # 1 XOR 0 = 1
        [np.array([[1]]), np.array([[1]])], # 1 XOR 1 = 0
    ]

    targets = [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[1]]),
        np.array([[0]]),
    ]
    
    rnn.initialiseWeights(inputSize=1, outputSize=1)

    initialLoss = rnn.train(inputs, targets)

    for i in range(100):
        loss = rnn.train(inputs, targets)
    
    assert loss < initialLoss