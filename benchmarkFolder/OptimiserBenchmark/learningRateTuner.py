from quacknet import GD, SGD, Adam, AdamW, Lion, RMSProp
from quacknet import CNNModel, Network
from quacknet import Conv2DLayer, PoolingLayer, ActivationLayer, DenseLayer, GlobalAveragePooling
import numpy as np
import time

images = np.load('benchmarkFolder/OptimiserBenchmark/data/train_images.npy').astype(np.float32)
labels = np.load('benchmarkFolder/OptimiserBenchmark/data/train_labels.npy').astype(np.float32)

learningRates = {
    "GD": np.logspace(-5, -2, 6),       # 1e-5 → 1e-2
    "SGD": np.logspace(-3, -1, 6),      # 1e-3 → 1e-1
    "Adam": np.logspace(-5, -2, 6),     # 1e-5 → 1e-2
    "AdamW": np.logspace(-5, -2, 6),    # 1e-5 → 1e-2
    "Lion": np.logspace(-4, -1.5, 6),   # 1e-4 → 3e-2
    "RMSProp": np.logspace(-5, -2, 6),  # 1e-5 → 1e-2
}
optimisationFunctions = {
    "GD": GD,
    "Adam": Adam,
    "AdamW": AdamW,
    "Lion": Lion,
    "RMSProp": RMSProp,
    "SGD": SGD,
}
optFuncResults = {
    "GD": [],
    "Adam": [],
    "AdamW": [],
    "Lion": [],
    "RMSProp": [],
    "SGD": [],
}

nn = Network(lossFunc="cross", learningRate=None, useBatches=True, batchSize=64, optimisationFunction=lambda x,y: ValueError("Not implemented"))
nn.addLayer(64)
nn.addLayer(10, "softmax")
cnn = CNNModel(nn, optimisationFunction=lambda x,y: ValueError("no optimisation function"))
depth = [3, 32]
numFilters = [32, 64]
for i in range(2):
    cnn.addLayer(Conv2DLayer(kernalSize=3, depth=depth[i], numKernals=numFilters[i], stride=2, padding="no"))
    cnn.addLayer(ActivationLayer())
    cnn.addLayer(PoolingLayer(gridSize=2, stride=2, mode="max"))
cnn.addLayer(GlobalAveragePooling())
cnn.addLayer(DenseLayer(nn))

nn.createWeightsAndBiases()
cnn.createWeightsBiases()

cnn.saveModel(nn.weights, nn.biases, cnn.weights, cnn.biases, "benchmarkFolder/OptimiserBenchmark/lrSweepParams.npz")

def resetParameters(nn, cnn):
    cnn.loadModel(nn, "benchmarkFolder/OptimiserBenchmark/lrSweepParams.npz")

for name, optFunc in optimisationFunctions.items():
    results = []
    for lr in learningRates:
        start = time.time()
        resetParameters(nn, cnn)

        nn.optimisationFunction = optFunc(cnn.forward, cnn._backpropagation)
        cnn.optimisationFunction = optFunc(cnn.forward, cnn._backpropagation)
        nn.learningRate = lr

        for i in range(3):
            _, loss = cnn.train(images, labels, True, 64, lr)
        results.append(float(loss))

        print(f"{name}, took: {time.time() - start}")
    optFuncResults[name] = results
    print(f"{name}: {results}")

print("")
for name, results in optFuncResults.items():
    print(f"{name}: {results}")
