import numpy as np
from quacknet import Network

def test_Weights():
    nn = Network()
    nn.addLayer(100)
    nn.addLayer(50, "relu")
    nn.addLayer(30, "sigmoid")

    nn.createWeightsAndBiases()

    assert len(nn.weights) == 2
    assert len(nn.biases) == 2

    assert nn.weights[0].shape == (100, 50)
    assert nn.weights[1].shape == (50, 30)
    assert nn.biases[0].shape == (50,)
    assert nn.biases[1].shape == (30,)

    for w in nn.weights:
        assert abs(np.mean(w)) < 0.2
    for b in nn.biases:
        assert abs(np.mean(b)) < 0.2