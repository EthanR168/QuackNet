import math

def ReLUDerivative(value):
    if(value > 0):
        return 1
    return 0

def sigmoid(value):
    return (1 / (1 + math.exp(-value)))

def SigmoidDerivative(value):
    return sigmoid(value) * (1 - sigmoid(value))

def TanHDerivative(value):
    def tanH(value):
        return ((math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value)))
    return 1 - (tanH(value) ** 2)

def LinearDerivative(_):
    return 1

def SoftMaxDerivative(currValueIndex, trueValue, values, lossDerivative):
    from lossDerivativeFunctions import CrossEntropyLossDerivative
    if(lossDerivative == CrossEntropyLossDerivative):
        return values[currValueIndex] - trueValue
    summ = 0
    for i in range(len(values)):
        if(currValueIndex == i):
            jacobianMatrix = values[currValueIndex] * (1 - values[currValueIndex])
        else:
            jacobianMatrix = -1 * values[currValueIndex] * values[i]
        summ += lossDerivative(values[i], trueValue[i], len(values)) * jacobianMatrix
    return summ
