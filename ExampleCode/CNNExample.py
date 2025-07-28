from quacknet.CNN.convulationalManager import CNNModel, ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from quacknet.main import Network
import numpy as np

# Creating parameters for convulational layer
inputTensor = [np.random.randn(4, 4, 4), np.random.randn(4, 4, 4)] #[(depth, height, width), (depth, height, width)]
trueLabels = np.array([[1], [1]])
learningRate = 0.001

# Define the dense layer
net = Network()  
net.addLayer(3) # Input layer with 3 neurons
net.addLayer(1) # Output layer
net.createWeightsAndBiases()

# Create the CNN model and add layers
CNN = CNNModel(net)
CNN.addLayer(ConvLayer(kernalSize = 2, depth = 4, numKernals = 3, stride = 2, padding = "0"))
CNN.addLayer(ActivationLayer()) # Leaky ReLU
CNN.addLayer(PoolingLayer(gridSize = None, stride = None, mode = "gap")) # Global Average Pooling
CNN.addLayer(DenseLayer(net)) # Fully connected Layer

CNN.createWeightsBiases()

# Train the model
accuracy, loss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

print(f"average accauracy: {accuracy:.2f}%")
print(f"average loss: {loss:.4f}")