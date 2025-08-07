from quacknet import StackedRNN
import numpy as np

def test_Stacked_CheckIfModelCanLearnOnSimpleData():
    np.random.seed(1028374)
    rnn = StackedRNN(
        hiddenStateActivationFunction="tanh",
        outputLayerActivationFunction="sigmoid",
        lossFunction="mse",
        numberOfHiddenStates=2,
        hiddenSizes=[32, 32],
        useBatches=False,
        batchSize=1, 
    )

    inputs = [ # Simple XOR
        [np.array([0]), np.array([0])], # 0 XOR 0 = 0
        [np.array([0]), np.array([1])], # 0 XOR 1 = 1
        [np.array([1]), np.array([0])], # 1 XOR 0 = 1
        [np.array([1]), np.array([1])], # 1 XOR 1 = 0
    ]

    targets = [
        np.array([0]),
        np.array([1]),
        np.array([1]),
        np.array([0]),
    ]
    
    rnn.initialiseWeights(inputSize=1, outputSize=1)

    initialLoss = rnn.train(inputs, targets, alpha=0.0001)

    for i in range(200):
        loss = rnn.train(inputs, targets, alpha=0.0001)
    
    assert loss < initialLoss