import math
import numpy as np

def relu(value):
    return max(0, value)
    
def sigmoid(value):
    return (1 / (1 + math.exp(-value)))
    
def tanH(value):
    return ((math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value)))
    
def linear(value):
    return value

def softMax(values): 
    summ = 0
    out = []
    for i in values:
        summ += math.exp(i)
    for i in values:
        out.append(math.exp(i)/summ)
    return np.array(out)
