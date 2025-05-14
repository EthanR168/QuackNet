import numpy as np

# Load the preprocessed data
train_images = np.load('MNISTExample/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('MNISTExample/train_labels.npy')  # Shape: (60000, 10)
test_images = np.load('MNISTExample/test_images.npy')   # Shape: (10000, 784)
test_labels = np.load('MNISTExample/test_labels.npy')    # Shape: (10000, 10)

from neuralLibrary.main import Network
import time

n = Network()
n.addLayer(784, "relu")
n.addLayer(64, "relu")
n.addLayer(10, "softmax")
n.createWeightsAndBiases()

print("started")
start = time.time()
n.train(train_images, train_labels , 1)
print(f"took: {time.time() - start}, took on average: {(time.time() - start)/len(train_images)} seconds")
