from quacknet import Network, CNNModel, Conv2DLayer, DenseLayer, GlobalAveragePooling, ActivationLayer
from quacknet import Adam
import numpy as np

def test_CNN_CheckIfModelCanLearnOnSimpleData():
    np.random.seed(2)

    batchSize = 3
    channels = 2
    height = width = 4

    inputTensor = np.random.randn(batchSize, channels, height, width)
    trueLabels = np.array([[1, 0], [0, 1], [0, 0]])

    learningRate = 0.001

    net = Network()  
    net.addLayer(3) 
    net.addLayer(2)
    net.createWeightsAndBiases()

    CNN = CNNModel(net, Adam)
    CNN.addLayer(Conv2DLayer(kernalSize=2, depth=channels, numKernals=3, stride=2, padding="no"))
    CNN.addLayer(ActivationLayer())
    CNN.addLayer(GlobalAveragePooling())
    CNN.addLayer(DenseLayer(net))
    CNN.createWeightsBiases()       

    _, initialLoss = CNN.train(inputTensor, trueLabels, useBatches = True, batchSize = batchSize, learningRate = learningRate)

    for _ in range(50):
        _, loss = CNN.train(inputTensor, trueLabels, useBatches = True, batchSize = batchSize, learningRate = learningRate)

    assert loss < initialLoss