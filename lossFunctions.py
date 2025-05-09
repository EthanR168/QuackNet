import math
import numpy as np

def MSELossFunction(predicted, true):
    #summ = 0
    #for i in range(len(predicted)):
    #    summ += (true[i] - predicted[i]) ** 2
    #return summ / len(predicted)
    return np.mean((np.array(true) - np.array(predicted)) ** 2)

def MAELossFunction(predicted, true):
    #summ = 0
    #for i in range(len(predicted)):
    #    summ += abs(true[i] - predicted[i])
    #return summ / len(predicted)
    return np.mean(np.abs(np.array(true) - np.array(predicted)))

def CrossEntropyLossFunction(predicted, true):
    #summ = 0
    #for i in range(len(predicted)):
    #    summ += true[i] * math.log(predicted[i])
    #return -summ
    return -np.sum(np.array(true) * np.log(predicted)) / len(predicted)
