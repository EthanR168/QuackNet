import numpy as np
import time

# Load the preprocessed data
train_images = np.load('ExampleCode/MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('ExampleCode/MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)

#test_images = np.load('ExampleCode/MNISTExample/data/test_images.npy')   # Shape: (10000, 784)
#test_labels = np.load('ExampleCode/MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from quacknet.main import Network

def run(epochs, steps, skipInput):
    learningRate = 0.01
    n = Network(
        learningRate = learningRate,
        lossFunc = "Cross Entropy",
        optimisationFunc = "batches", # Gradient Descent
        useBatches = True,
        batchSize = 64
    )
    
    n.addLayer(784, "relu")
    n.addLayer(128, "relu")
    n.addLayer(64, "relu")
    n.addLayer(10, "softmax")

    if(skipInput == True):
        inp = input("Create new weights/biases (y/n): ").lower()
    else:
        inp = "y"

    if(inp == "y"):
        n.createWeightsAndBiases()
        n.write() # writes weights/biases to a file
    else:
        n.read()

    accuracies, losses = [], []
    for epoch in range(0, epochs, steps):
        start = time.time()
        accuaracy, averageLoss = n.train(train_images, train_labels, steps)
        print(f"epoch: {steps * (epoch + 1)} / {epochs*steps}, took: {(time.time() - start)} seconds, accuracy: {round(accuaracy*100,2)}%, average loss: {averageLoss}")
        n.write() # writes weights/biases to a file
        accuracies.append(accuaracy)
        losses.append(averageLoss)
    totalAccuracy.append(accuracies)
    totalLoss.append(losses)

    n.write() # writes weights/biases to a file

totalAccuracy, totalLoss = [], []
for _ in range(5):
    run(epochs = 10, steps = 1, skipInput = False)

Network.drawGraphs(None, totalAccuracy, totalLoss) # draws accuracy and loss graphs 