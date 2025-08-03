from quacknet import SingularRNN
import numpy as np

rnn = SingularRNN(hiddenStateActivationFunction="tanh", outputLayerActivationFunction="sigmoid", lossFunction="mse")

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

learningRate = 0.01
epochs = 500

rnn.initialiseWeights(inputSize=1, hiddenSize=64, outputSize=1)
for i in range(epochs):
    loss = rnn.train(inputs, targets)
    if((i + 1) % 20 == 0 or i == 0):
        print(f"Loss {i + 1}: {loss}")