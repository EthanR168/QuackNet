import numpy as np
from neuralLibrary.main import Network 
from neuralLibrary.activationFunctions import relu

class TestNetwork:
    def test_calculateLayerNodesTest1(self):
        currLayer = np.array([1, relu])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.2 + 0.25 * 0.75 + 0.5 * 0.5
        assert Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer) == np.array([0.6375])

    def test_calculateLayerNodesTest2(self):
        currLayer = np.array([1, relu])
        lastLayerNodes = np.array([1, 0.25])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.5])
        #0.5 + 1 * 0.75 + 0.25 * 0.5
        assert Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer) == np.array([1.375])

    def test_forwardPropogation(self):
        n = Network()
        n.addLayer(2)
        n.addLayer(1)
        n.weights = np.array([[0.75, 0.5]])
        inputLayer = [0.25, 0.5]
        n.biases = np.array([0.2])
        resulting = n.forwardPropagation(inputLayer)
        assert np.allclose(resulting[0], np.array([0.25, 0.5]), np.array([0.6375]))
        assert np.allclose(resulting[1], np.array([0.6375]))