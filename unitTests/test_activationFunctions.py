import pytest
import numpy as np
from neuralLibrary.activationFunctions import relu, sigmoid, tanH, linear, softMax

def test_relu():
    assert relu(-0.3) == 0
    assert relu(0) == 0
    assert relu(0.4) == 0.4

def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(0.4) == pytest.approx(0.598687660112)
    assert sigmoid(-0.4) == pytest.approx(0.401312339888)

def test_tanH():
    assert tanH(0) == 0
    assert tanH(0.5) == pytest.approx(0.46211715726)
    assert tanH(-0.5) == pytest.approx(-0.46211715726)

def test_linear():
    assert linear(0) == 0
    assert linear(0.6) == 0.6
    assert linear(-0.5) == -0.5

def test_softMax():
    assert np.allclose(softMax(np.array([0.5, 0.5])), np.array([0.5, 0.5]))
    assert np.allclose(softMax([0.25, 0.5, 0.25]), np.array([0.30450434242, 0.390991315159, 0.30450434242]))
