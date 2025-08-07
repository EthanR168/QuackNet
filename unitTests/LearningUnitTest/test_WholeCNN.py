from quacknet import Network, CNNModel, Conv2DLayer, DenseLayer, GlobalAveragePooling, ActivationLayer
import numpy as np

def test_CNN_CheckIfModelCanLearnOnSimpleData():
    np.random.seed(2)

    batch_size = 2
    channels = 2
    height = width = 4

    inputTensor = np.random.randn(batch_size, channels, height, width)
    trueLabels = np.array([[1], [1]])

    learningRate = 0.001

    net = Network()  
    net.addLayer(3) 
    net.addLayer(1)
    net.createWeightsAndBiases()

    CNN = CNNModel(net)
    CNN.addLayer(Conv2DLayer(kernalSize=2, depth=channels, numKernals=3, stride=2, padding="no"))
    CNN.addLayer(ActivationLayer())
    CNN.addLayer(GlobalAveragePooling())
    CNN.addLayer(DenseLayer(net))
    CNN.createWeightsBiases()       

    _, initialLoss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

    for _ in range(50):
        _, loss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

    
    assert loss < initialLoss