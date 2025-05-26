from quacknet.main import Network

# Define a neural network architecture
n = Network(
    lossFunc = "cross entropy",
    learningRate = 0.01,
    optimisationFunc = "sgd", #stochastic gradient descent
)
n.addLayer(3, "relu") # Input layer
n.addLayer(2, "relu") # Hidden layer
n.addLayer(1, "softmax") # Output layer
n.createWeightsAndBiases()

# Example data
inputData = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
labels = [[1], [0]]

# Train the network
accuracy, averageLoss = n.train(inputData, labels, epochs = 10)

# Evaluate
print(f"Accuracy: {accuracy}%")
print(f"Average loss: {averageLoss}")