import math
import numpy as np

def MSELossFunction(predicted, true):
    return np.mean((np.array(true) - np.array(predicted)) ** 2)

def MAELossFunction(predicted, true):
    return np.mean(np.abs(np.array(true) - np.array(predicted)))

def CrossEntropyLossFunction(predicted, true):
    return -np.sum(np.array(true) * np.log(predicted))
