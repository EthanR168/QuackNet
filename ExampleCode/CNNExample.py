from neuralLibrary.convulationalManager import CNNModel, ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from neuralLibrary.main import Network
import numpy as np

# Creating parameters for convulational layer
inputTensor = np.random.randn(4, 4, 4) #(depth, height, width)
kernalWeights = np.random.randn(3, 4, 2, 2)
kernalBiases = np.random.randn(3)

# Define the dense layer
net = Network()  
net.addLayer(48)
net.addLayer(1)
net.createWeightsAndBiases()

# Define the CNN model
CNN = CNNModel()
CNN.addLayer(ConvLayer(kernalWeights.shape[2], kernalWeights, kernalBiases, len(kernalWeights), 1, padding = "1"))
CNN.addLayer(ActivationLayer())
CNN.addLayer(PoolingLayer(2, 1, "max"))
CNN.addLayer(DenseLayer(net))

# outputs a 1d array due to the dense layer
output = CNN.forward(inputTensor)

print("input tensor:")
print(inputTensor)
print("output tensor:")
print(output)

print(f"input shape: {inputTensor.shape}")
print(f"output shape: {output.shape}")