from neuralLibrary.convulationalManager import CNNModel, ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from neuralLibrary.main import Network
import numpy as np

# Creating parameters for convulational layer
inputTensor = [np.random.randn(4, 4, 4), np.random.randn(4, 4, 4)] #[(depth, height, width), (depth, height, width)]
trueValues = np.array([[1], [1]])

# Define the dense layer
net = Network()  
net.addLayer(27)
net.addLayer(1)
net.createWeightsAndBiases()

# Define the CNN model
CNN = CNNModel(net)
CNN.addLayer(ConvLayer(2, 4, 3, 2, padding = "1"))
CNN.addLayer(ActivationLayer())
CNN.addLayer(PoolingLayer(2, 1, "max"))
CNN.addLayer(DenseLayer(net))

# Creates weights and biases 
CNN.createWeightsBiases()

accuaracy, loss = CNN.train(inputTensor, trueValues, False, 1)

print(f"average accauracy: {accuaracy}%")
print(f"average loss: {loss}")