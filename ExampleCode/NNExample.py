from neuralLibrary.main import Network
import numpy as np

n = Network()
n.addLayer(3)
n.addLayer(2)
n.addLayer(1)
n.createWeightsAndBiases()
n.train([[0.5, 0.5, 0.5]], np.array([1]), 1)