from quacknet import Network, SGD

# Define a basic neural network architecture
n = Network(
    lossFunc = "cross entropy",
    learningRate = 0.01,
    optimisationFunction = SGD, # Stochastic Gradient Descent
    useBatches=False,
)
n.addLayer(5, "relu") # Input layer
n.addLayer(4, "relu") # Hidden layer
n.addLayer(3, "softmax") # Output layer

n.createWeightsAndBiases()

# Example training data
inputData = [[0.1, 0.2, 0.3, 0.4, 0.5]]
labels = [[1, 0, 0]]
numEpochs = 1

# Train the network
accuracy, averageLoss = n.train(inputData, labels, numEpochs)

# Evaluate
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average loss: {averageLoss:.4f}")