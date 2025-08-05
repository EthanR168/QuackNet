from quacknet import Network, CNNModel, Conv2DLayer, DenseLayer, GlobalAveragePooling, ActivationLayer
import numpy as np

def test_CNN_CheckIfModelCanLearnOnSimpleData():
    inputTensor = [np.random.randn(4, 4, 4), np.random.randn(4, 4, 4)] #[(depth, height, width), (depth, height, width)]
    trueLabels = np.array([[1], [1]])
    learningRate = 0.001

    net = Network()  
    net.addLayer(3) 
    net.addLayer(1)
    net.createWeightsAndBiases()

    CNN = CNNModel(net)
    CNN.addLayer(Conv2DLayer(kernalSize = 2, depth = 4, numKernals = 3, stride = 2, padding = "0"))
    CNN.addLayer(ActivationLayer())      
    CNN.addLayer(GlobalAveragePooling()) 
    CNN.addLayer(DenseLayer(net))        

    CNN.createWeightsBiases()

    _, initialLoss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

    for _ in range(10):
        _, loss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

    
    assert loss < initialLoss