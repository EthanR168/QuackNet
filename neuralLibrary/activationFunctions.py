import numpy as np

def relu(values):
    return np.maximum(0, values)
     
def sigmoid(values):
    return 1 / (1 + np.exp(-values))

def tanH(values):
    return np.tanh(values)

def linear(values): #Dont use too demanding on CPU
    return values

def softMax(values): 
    summ = np.sum(np.exp(values))
    out = np.exp(values) / summ
    return out
