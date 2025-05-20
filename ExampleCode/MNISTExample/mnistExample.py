import numpy as np
import time

# Load the preprocessed data
train_images = np.load('ExampleCode/MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('ExampleCode/MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)
#test_images = np.load('ExampleCode/MNISTExample/data/test_images.npy')   # Shape: (10000, 784)
#test_labels = np.load('ExampleCode/MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from neuralLibrary.main import Network

def run(epochs):
    learningRate = 0.005
    n = Network(learningRate=learningRate, lossFunc="cross", optimisationFunc="batches", useBatches=True, batchSize=64, useMomentum=True)
    n.addLayer(784, "relu")
    n.addLayer(128, "relu")
    n.addLayer(64, "relu")
    n.addLayer(10, "softmax")

    inp = "y" #input("Create new weights/biases (y/n): ").lower()
    if(inp == "y"):
        n.createWeightsAndBiases()
        n.write()
    else:
        n.read()

    for epoch in range(epochs):
        print("started")
        start = time.time()
        accuaracy, averageLoss = n.train(train_images, train_labels, 1)
        print(f"epoch: {1 * (epoch + 1)}/{epochs*1}, took: {(time.time() - start)} seconds, accuracy: {round(accuaracy*100,2)}%, average loss: {averageLoss}")
        n.write()
    
    n.write()

run(5)