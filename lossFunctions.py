import math
import numpy as np

def MSELossFunction(predicted, true):
    #summ = 0
    #for i in range(len(predicted)):
    #    summ += (true[i] - predicted[i]) ** 2
    #return summ / len(predicted)
    return np.mean((np.array(true) - np.array(predicted)) ** 2)

def MAELossFunction(predicted, true):
    summ = 0
    for i in range(len(predicted)):
        summ += abs(true[i] - predicted[i])
    return summ / len(predicted)

def CrossEntropyLossFunction(predicted, true):
    summ = 0
    for i in range(len(predicted)):
        summ += true[i] * math.log(predicted[i])
    return -summ
