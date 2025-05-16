import numpy as np
import time

# Load the preprocessed data
train_images = np.load('MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)
#test_images = np.load('MNISTExample/data/test_images.npy')   # Shape: (10000, 784)
#test_labels = np.load('MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from neuralLibrary.main import Network

def run(epochs):
    n = Network(learningRate=1, lossFunc="cross")
    n.addLayer(784, "relu")
    n.addLayer(128, "relu")
    n.addLayer(128, "relu")
    n.addLayer(10, "softmax")

    n.createWeightsAndBiases()
    n.write()

    for _ in range(epochs):
        print("started")
        start = time.time()

        accuaracy = n.train(train_images[0:1], train_labels[0:1] , 1)
        print(f"took: {(time.time() - start)} seconds, accuracy: {accuaracy}")
    
    n.write()

run(100)