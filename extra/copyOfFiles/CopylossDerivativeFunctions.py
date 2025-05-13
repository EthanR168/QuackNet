from activationDerivativeFunctions import SoftMaxDerivative

def MSEDerivative(value, trueValue, sizeOfLayer):
    return 2 * (trueValue - value) / sizeOfLayer

def MAEDerivative(value, trueValue, sizeOfLayer):
    summ = value - trueValue
    if(summ > 0):
        return 1 / sizeOfLayer
    elif(summ < 0):
        return -1 / sizeOfLayer
    return 0

def CrossEntropyLossDerivative(value, trueVale, activationDerivative):
    if(activationDerivative == SoftMaxDerivative):
        return value - trueVale
    return -1 * (trueVale / value)
