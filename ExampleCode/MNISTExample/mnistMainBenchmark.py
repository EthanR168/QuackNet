import numpy as np

# Load the preprocessed data
train_images = np.load('ExampleCode/MNISTExample/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('ExampleCode/MNISTExample/data/train_labels.npy')  # Shape: (60000, 10)
test_images = np.load('ExampleCode/MNISTExample/data/test_images.npy')    # Shape: (10000, 784)
test_labels = np.load('ExampleCode/MNISTExample/data/test_labels.npy')    # Shape: (10000, 10)

from quacknet import Network
import time

def benchmark(size):
    aves = []
    for i in range(5):
        n = Network(lossFunc="cross")
        n.addLayer(784, "relu")
        n.addLayer(128, "relu")
        n.addLayer(128, "relu")
        n.addLayer(10, "softmax")
        n.createWeightsAndBiases()

        inputs = train_images[0:size]
        labels = train_labels[0:size]

        print("started")
        start = time.time()
        n.train(inputs, labels, 1)
        ave = (time.time() - start)/size
        print(f"for size: {size}, took on average: {ave} seconds")
        aves.append(ave)
    print(f"for size: {size}, took on average across 5 epochs: {np.sum(aves)/5} seconds")
    

benchmark(1)
benchmark(10)
benchmark(100)
benchmark(1000)
benchmark(10000)
benchmark(20000)
benchmark(60000)

# use to identify bottlenecks: 
# python -m cProfile -s tottime ExampleCode/MNISTExample/mnistMainBenchmark.py 