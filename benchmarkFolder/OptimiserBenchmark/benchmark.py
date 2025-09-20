from quacknet import GD, Adam, Lion, RMSProp
from quacknet import CNNModel, Network
from quacknet import Conv2DLayer, PoolingLayer, ActivationLayer, DenseLayer, GlobalAveragePooling
import numpy as np
import time

images = np.load('benchmarkFolder/OptimiserBenchmark/data/train_images.npy').astype(np.float32)[0:10000]
labels = np.load('benchmarkFolder/OptimiserBenchmark/data/train_labels.npy').astype(np.float32)[0:10000]

learningRates = {
    "GD": 1e-4,         
    "Adam": 1e-3,    
    "Lion": 5e-3,    
    "RMSProp": 1e-3,
}
optimisationFunctions = {
    "GD": GD,
    "Adam": Adam,
    "Lion": Lion,
    "RMSProp": RMSProp,
}
optFuncResults_Loss = {
    "GD": [],
    "Adam": [],
    "Lion": [],
    "RMSProp": [],
}
optFuncResults_Accuracy = {
    "GD": [],
    "Adam": [],
    "Lion": [],
    "RMSProp": [],
}

nn = Network(lossFunc="normalised cross", learningRate=None, useBatches=True, batchSize=64, optimisationFunction=lambda x,y: ValueError("Not implemented"))
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

def loadParameters(nn, cnn):
    cnn.loadModel(nn, "benchmarkFolder/OptimiserBenchmark/initialParameters.npz")

numEpochs = 10

for name, optFunc in optimisationFunctions.items():
    results_Loss = []
    results_Accuracy = []
    loadParameters(nn, cnn)
    for epoch in range(numEpochs):
        start = time.time()

        nn.optimisationFunction = optFunc(cnn.forward, cnn._backpropagation)
        cnn.optimisationFunction = optFunc(cnn.forward, cnn._backpropagation)
        nn.learningRate = learningRates[name]

        acc, loss = cnn.train(images, labels, True, 64, learningRates[name])

        results_Loss.append(float(loss))
        results_Accuracy.append(float(acc))

        print(f"{name}, epoch: {epoch+1} / {numEpochs}, took: {time.time() - start}")

    optFuncResults_Loss[name] = results_Loss
    optFuncResults_Accuracy[name] = results_Accuracy
    print(f"{name}: {results_Accuracy}")
    print(f"{name}: {results_Loss}")

print("")
print("Loss:")
for name, results in optFuncResults_Loss.items():
    print(f"{name}: {results}")

print("")
print("Accuracy:")
for name, results in optFuncResults_Accuracy.items():
    print(f"{name}: {results}")

"""
optFuncResults_Loss = {
    "GD": [],
    "Adam": [],
    "Lion": [],
    "RMSProp": [],
}
optFuncResults_Accuracy = {
    "GD": [],
    "Adam": [],
    "Lion": [],
    "RMSProp": [],
}
"""