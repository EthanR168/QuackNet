import numpy as np
from quacknet.lossDerivativeFunctions import MSEDerivative, MAEDerivative, CrossEntropyLossDerivative
from quacknet.activationDerivativeFunctions import SoftMaxDerivative

def test_MSELossDerivative():
    assert np.allclose(MSEDerivative(np.array([0.5, 0.5]), np.array([1, 0]), 2), np.array([-0.5, 0.5]))
    assert np.allclose(MSEDerivative(np.array([1, 1]), np.array([1, 1]), 2),  np.array([0, 0]))
    assert np.allclose(MSEDerivative(np.array([0]), np.array([1]), 1), np.array([-2]))
    assert np.allclose(MSEDerivative(np.array([1e6]), np.array([0]), 1), 2e6)

def test_MAELossDerivative():
    assert np.allclose(MAEDerivative(np.array([0.5, 0.5]), np.array([1, 0]), 2), np.array([-0.5, 0.5]))
    assert np.allclose(MAEDerivative(np.array([1, 1]), np.array([1, 1]), 2), np.array([0, 0]))
    assert np.allclose(MAEDerivative(np.array([0]), np.array([1]), 1), np.array([-1]))
    assert np.allclose(MSEDerivative(np.array([1e6]), np.array([0]), 1), np.array([2e6]))

def test_CrossEntropyLossDerivative_NotUsingSoftmax():
    assert np.allclose(CrossEntropyLossDerivative(np.array([1, 1]), np.array([0.5, 0.5]), None), np.array([-0.5, -0.5]))
    assert np.allclose(CrossEntropyLossDerivative(np.array([0.000001]), np.array([1]), None), np.array([-1e6]))
    assert np.allclose(CrossEntropyLossDerivative(np.array([0.1, 0.8, 0.1]), np.array([0, 1, 0]), None), np.array([0, -1.25, 0]))

def test_CrossEntropyLossDerivative_UsingSoftmax(): 
    assert np.allclose(CrossEntropyLossDerivative(np.array([1, 1]), np.array([0.5, 0.5]), SoftMaxDerivative), np.array([0.5, 0.5]))
    assert np.allclose(CrossEntropyLossDerivative(np.array([0]), np.array([1]), SoftMaxDerivative), np.array([-1]))
    assert np.allclose(CrossEntropyLossDerivative(np.array([0.1, 0.8, 0.1]), np.array([0, 1, 0]), SoftMaxDerivative), np.array([0.1, -0.2, 0.1]))