from quacknet import CNNModel, Conv2DLayer, GlobalAveragePooling, DenseLayer, ActivationLayer, Network
import numpy as np

# Creating parameters for convulational layer
batch_size = 2
channels = 2
height = width = 4
inputTensor = np.random.randn(batch_size, channels, height, width)
trueLabels = np.array([[1], [1]])

learningRate = 0.001

# Define the dense layer
net = Network()  
net.addLayer(3) # Input layer with 3 neurons
net.addLayer(1) # Output layer
net.createWeightsAndBiases()

# Create the CNN model and add layers
CNN = CNNModel(net)
CNN.addLayer(Conv2DLayer(kernalSize=2, depth=channels, numKernals=3, stride=2, padding="no"))
CNN.addLayer(ActivationLayer())      # Leaky ReLU
CNN.addLayer(GlobalAveragePooling()) # Global Average Pooling
CNN.addLayer(DenseLayer(net))        # Fully connected Layer
CNN.createWeightsBiases()

# Train the model
accuracy, loss = CNN.train(inputTensor, trueLabels, useBatches = False, batchSize = None, alpha = learningRate)

print(f"average accauracy: {accuracy:.2f}%")
print(f"average loss: {loss:.4f}")