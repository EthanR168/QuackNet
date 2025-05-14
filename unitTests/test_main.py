import numpy as np
from neuralLibrary.main import Network 
from neuralLibrary.activationFunctions import relu, sigmoid, tanH, linear, softMax

class TestNetwork_CalculateLayerNodes:
    def test_calculateLayerNodes_Relu(self):
        currLayer = np.array([1, relu])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.2 + 0.25 * 0.75 + 0.5 * 0.5
        assert Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer) == np.array([0.6375])

        currLayer = np.array([3, relu])
        lastLayerNodes = np.array([0.25, 0.5, 0.1])
        lastLayerWeights = np.array([[0.75], [0.5], [0.25]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.1 * 0.25 + 0.2
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.6625]))

    def test_calculateLayerNodes_sigmoid(self):
        currLayer = np.array([2, sigmoid])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        #sigmoid(0.6375) == 0.654188113761
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.654188113761]))

        currLayer = np.array([3, sigmoid])
        lastLayerNodes = np.array([0.25, 0.5, 0.1])
        lastLayerWeights = np.array([[0.75], [0.5], [0.25]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.1 * 0.25 + 0.2 = 0.6625
        #sigmoid(0.6625) == 0.659821754972
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.659821754972]))

    def test_calculateLayerNodes_tanH(self):
        currLayer = np.array([2, tanH])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        #tanH(0.6375) == 0.563194927805
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.563194927805]))

        currLayer = np.array([3, tanH])
        lastLayerNodes = np.array([0.25, 0.5, 0.1])
        lastLayerWeights = np.array([[0.75], [0.5], [0.25]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.1 * 0.25 + 0.2 = 0.6625
        #tanH(0.6625) == 0.659821754972
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.580024746853]))

    def test_calculateLayerNodes_linear(self):
        currLayer = np.array([2, linear])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.6375]))

        currLayer = np.array([3, linear])
        lastLayerNodes = np.array([0.25, 0.5, 0.1])
        lastLayerWeights = np.array([[0.75], [0.5], [0.25]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.1 * 0.25 + 0.2 = 0.6625
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([0.6625]))

    def test_calculateLayerNodes_softmax(self):
        currLayer = np.array([2, softMax])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75], [0.5]])
        biases = np.array([0.2])
        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        assert np.allclose(Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer), np.array([1]))

        currLayer = np.array([2, softMax])
        lastLayerNodes = np.array([0.25, 0.5])
        lastLayerWeights = np.array([[0.75, 0.25], [0.5, 0.75]])
        biases = np.array([0.2, 0.3])

        out1 = 0.6375 # 0.25 * 0.75 + 0.5 * 0.5 + 0.2
        out2 = 0.7375 # 0.25 * 0.25 + 0.5 * 0.75 + 0.3

        summ = np.exp(out1) + np.exp(out2)
        s1 = np.exp(out1) / summ
        s2 = np.exp(out2) / summ
        result = Network.calculateLayerNodes(self, lastLayerNodes, lastLayerWeights, biases, currLayer)
        assert np.allclose(s1, result[0])
        assert np.allclose(s2, result[1])

class TestNetwork_ForwardPropogation:
    def test_forwardPropogation_noHiddenLayer(self):
        n = Network()
        n.addLayer(2)
        n.addLayer(1)
        n.weights = np.array([[0.75, 0.5]])
        inputLayer = [0.25, 0.5]
        n.biases = np.array([0.2])
        resulting = n.forwardPropagation(inputLayer)
        assert np.allclose(resulting[0], np.array([0.25, 0.5]))
        assert np.allclose(resulting[1], np.array([0.6375]))

    def test_forwardPropogation_withHiddenLayer(self):
        n = Network()
        n.addLayer(2)
        n.addLayer(3, "relu")
        n.addLayer(2, "softmax")
        n.weights = [
            np.array([[0.75, 0.25, 0.1], [0.5, 0.75, 0.2]]),
            np.array([[0.5, 0.2], [0.4, 0.1], [0.3, 0.6]])
        ]
        n.biases = [
            np.array([0.2, 0.3, 0.1]),
            np.array([0.4, 0.5])
        ]
        inputLayer = [0.25, 0.5]

        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        #0.25 * 0.25 + 0.5 * 0.75 + 0.3 = 0.7375
        #0.25 * 0.1 + 0.5 * 0.2 + 0.1 = 0.225

        hidden = np.maximum(0, np.array([0.6375, 0.7375, 0.225]))
        out = np.dot(hidden, n.weights[1]) + n.biases[1]
        output = np.exp(out) / np.sum(np.exp(out))

        resulting = n.forwardPropagation(inputLayer)

        assert np.allclose(resulting[0], inputLayer)
        assert np.allclose(resulting[1], hidden)
        assert np.allclose(resulting[2], output)