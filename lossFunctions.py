import math

def MSELossFunction(predicted, true):
    summ = 0
    for i in range(len(predicted)):
        summ += (true[i] - predicted[i]) ** 2
    return summ / len(predicted)

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
