from neuralLibrary.main import Network
import time

n = Network()
n.addLayer(784, "relu")
n.addLayer(64, "relu")
n.addLayer(10, "softmax")
n.createWeightsAndBiases()

print("started")
start = time.time()
n.train([[0] * 784], [0] * 10, 1)
print(f"took: {time.time() - start} seconds")