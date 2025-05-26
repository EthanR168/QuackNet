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
    n = Network(learningRate=learningRate, lossFunc="cross", optimisationFunc="batches", useBatches=True, batchSize=64)
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
        n.write()
    else:
        n.read()

    accuracies, losses = [], []
    for epoch in range(0, epochs, steps):
        start = time.time()
        accuaracy, averageLoss = n.train(train_images, train_labels, steps)
        print(f"epoch: {steps * (epoch + 1)}/{epochs*steps}, took: {(time.time() - start)} seconds, accuracy: {round(accuaracy*100,2)}%, average loss: {averageLoss}")
        n.write()
        accuracies.append(accuaracy)
        losses.append(averageLoss)
    allAc.append(accuracies)
    allLos.append(losses)

    n.write()

allAc, allLos = [], []
for _ in range(5):
    run(10, 1, False)
Network.drawGraphs(None, allAc, allLos)