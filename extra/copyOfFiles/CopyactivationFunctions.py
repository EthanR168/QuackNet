import math
import numpy as np

def relu(values):
    v = []
    for i in values:
        v.append(max(0, i))
    return v
     
def sigmoid(values):
    v = []
    for val in values:
        v.append(1 / (1 + math.exp(-val)))
    return v

def tanH(values):
    v = []
    for val in values:
        v.append((math.exp(val) - math.exp(-val)) / (math.exp(val) + math.exp(-val)))
    return v

def linear(values):
    return values

def softMax(values): 
    summ = 0
    out = []
    for i in values:
        summ += math.exp(i)
    for i in values:
        out.append(math.exp(i)/summ)
    return np.array(out)
