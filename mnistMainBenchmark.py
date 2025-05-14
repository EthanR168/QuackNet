import numpy as np

# Load the preprocessed data
train_images = np.load('MNISTExample/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('MNISTExample/train_labels.npy')  # Shape: (60000, 10)
test_images = np.load('MNISTExample/test_images.npy')    # Shape: (10000, 784)
test_labels = np.load('MNISTExample/test_labels.npy')    # Shape: (10000, 10)

from neuralLibrary.main import Network
import time

def benchmark(size):
    n = Network()
    n.addLayer(784, "relu")
    n.addLayer(64, "relu")
    n.addLayer(10, "softmax")
    n.createWeightsAndBiases()

    inputs = train_images[0:size]
    labels = train_labels[0:size]

    print("started")
    start = time.time()
    n.train(inputs, labels, 1)
    print(f"for size: {size}, took on average: {(time.time() - start)/size} seconds")

benchmark(1)
benchmark(10)
benchmark(100)
benchmark(1000)
benchmark(10000)
benchmark(20000)
benchmark(60000)