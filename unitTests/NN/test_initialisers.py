from neuralLibrary.initialisers import Initialisers
from neuralLibrary.main import Network
import numpy as np

def test_initialiser():
    np.random.seed(43)
    layers = [10, 8, 6, 2]

    net = Network()

    for i in range(len(layers)):
        net.addLayer(layers[i])

    net.createWeightsAndBiases()

    weights, biases = [], []
    for i in range(1, len(layers)):
        weights.append(np.random.normal(0, np.sqrt(2 / layers[i - 1]), (layers[i - 1], layers[i])))
        biases.append(np.random.normal(0, np.sqrt(2 / layers[i - 1]), (layers[i])))
    
    assert np.array(weights[0]).shape == net.weights[0].shape
    assert np.array(weights[1]).shape == net.weights[1].shape
    assert np.array(weights[2]).shape == net.weights[2].shape

    assert np.array(biases[0]).shape == net.biases[0].shape
    assert np.array(biases[1]).shape == net.biases[1].shape
    assert np.array(biases[2]).shape == net.biases[2].shape
