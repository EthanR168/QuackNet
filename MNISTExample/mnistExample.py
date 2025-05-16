import numpy as np
import time

# Load the preprocessed data
train_images = np.load('MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)
#test_images = np.load('MNISTExample/data/test_images.npy')   # Shape: (10000, 784)
#test_labels = np.load('MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from neuralLibrary.main import Network

def run(epochs):
    n = Network(learningRate=0.001, lossFunc="cross")
    n.addLayer(784, "relu")
    n.addLayer(128, "relu")
    n.addLayer(128, "relu")
    n.addLayer(10, "softmax")

    inp = "n" #input("Create new weights/biases (y/n): ").lower()
    if(inp == "y"):
        n.createWeightsAndBiases()
        n.write()
    else:
        n.read()

    for _ in range(epochs):
        print("started")
        start = time.time()

        accuaracy = n.train(train_images, train_labels , 1)
        print(n.weights[0][0][0])
        print(f"took: {(time.time() - start)} seconds, accuracy: {accuaracy}")
    
    n.write()

run(10)