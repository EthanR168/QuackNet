import numpy as np
from neuralLibrary.lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction

def test_MSELossFunction():
    assert MSELossFunction(np.array([0.5, 0.5]), np.array([1, 0])) == 0.25
    assert MSELossFunction(np.array([1, 1]), np.array([1, 1])) == 0

def test_MAELossFunction():
    assert MAELossFunction(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert MAELossFunction(np.array([1, 1]), np.array([1, 1])) == 0
    assert MAELossFunction(np.array([0.75, 0.75]), np.array([1, 0.25])) == 0.375

def test_CrossEntropyLossFunction():
    assert CrossEntropyLossFunction(np.array([1, 1]), np.array([0.5, 0.5])) == 0
    assert CrossEntropyLossFunction(np.array([0.01]), np.array([1])) == -np.log(0.01)
    assert CrossEntropyLossFunction(np.array([0.1, 0.8, 0.1]), np.array([0, 1, 0])) == -np.log(0.8)