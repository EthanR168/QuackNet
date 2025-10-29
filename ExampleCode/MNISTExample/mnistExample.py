import numpy as np
import time

# Load the preprocessed data
train_images = np.load('ExampleCode/MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('ExampleCode/MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)

#test_images = np.load('ExampleCode/MNISTExample/data/test_images.npy')   # Shape: (10000, 784)
#test_labels = np.load('ExampleCode/MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from quacknet import Network, Adam
from quacknet import drawGraphs

def run(epochs):
    learningRate = 0.01
    n = Network(
        learningRate = learningRate,
        lossFunc = "MSE",
        optimisationFunction = Adam, 
        useBatches = True,
        batchSize = 64
    )
    
    n.addLayer(784, "linear") #"relu")
    n.addLayer(128, "linear") # "relu")
    n.addLayer(64, "linear") #  "relu")
    n.addLayer(10, "linear") #  "softmax")

    n.createWeightsAndBiases()
    n.write() # writes weights/biases to a file

    accuracies, losses = [], []
    for epoch in range(epochs):
        start = time.time()
        accuaracy, averageLoss = n.train(train_images, train_labels, 1)

        accuracies.append(accuaracy)
        losses.append(averageLoss)

        print(f"epoch: {epoch + 1} / {epochs}, took: {(time.time() - start)} seconds, accuracy: {round(accuaracy*100,2)}%, average loss: {averageLoss}")
        n.write() # writes weights/biases to a file

    totalAccuracy.append(accuracies)
    totalLoss.append(losses)

    n.write() # writes weights/biases to a file

totalAccuracy, totalLoss = [], []
for _ in range(5):
    run(10)

drawGraphs(None, totalAccuracy, totalLoss) # draws accuracy and loss graphs 