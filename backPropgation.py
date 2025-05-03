import math
import main

'''
output layer backpropogation for weights:
e = (dL/da) * f'(z)
e = error term
dL/da = derivative of the loss function
f'() = derivative of the activation function
z = the current layer's node (only one)

(dL/dW) = e * a
dL/dW  = derivative of loss function with respect to weight
e = error term
a = past layer's node value

nw = ow - r * (dL/dW)
nw = new weight
ow = old weight
r = learning rate
(dL/dW) = derivative of loss function with respect to weight
'''

def outputLayerWeightChange(lossDerivative, activationDerivative, currentLayerNodes, pastLayerNodes, pastLayerWeights, learningRate, trueValues):
    errorTerms = []
    weightGradients = []
    for i in range(len(currentLayerNodes)):
        lossDerivativeValue = lossDerivative(currentLayerNodes[i], trueValues[i], len(currentLayerNodes))
        if(lossDerivative == CrossEntropyLossDerivative):
            lossDerivativeValue = CrossEntropyLossDerivative(currentLayerNodes[i], trueValues[i], activationDerivative)

        if(activationDerivative == SoftMaxDerivative):
            errorTerm =  lossDerivativeValue *  activationDerivative(i, trueValues[i], currentLayerNodes, lossDerivative)
        else:
            errorTerm = lossDerivativeValue * activationDerivative(currentLayerNodes[i])
        for j in range(len(pastLayerNodes)):
            weightGradients.append(errorTerm * pastLayerNodes[j])
            #pastLayerWeights[j][i] -= learningRate * weightGradient 
        errorTerms.append(errorTerm)
    return weightGradients, errorTerms

'''
hidden layer backpropgation for weights:
e = SUM(e[l + 1][k] * w[l + 1][k]) * f'(z)
e = error term
SUM(e[l + 1][k] * w[l + 1][k]) = the sum of the next layers's error term for the current node multiplied by the weight in the nextlayer connected to the current one
f'() = derivative of the activation function
z = the current layer's node (only one)

(dL/dW) = e * a
dL/dW  = derivative of loss function with respect to weight
e = error term
a = past layer's node value

nw = ow - r * (dL/dW)
nw = new weight
ow = old weight
r = learning rate
(dL/dW) = derivative of loss function with respect to weight
'''

def hiddenLayerWeightChange(pastLayerErrorTerms, pastLayerWeights, activationDerivative, currentLayerNodes, pastLayerNodes, learningRate):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        errorTerm = 0
        for node in range(len(pastLayerNodes)):
            errorTerm += pastLayerErrorTerms[node] * pastLayerWeights[node][i]
        errorTerm = errorTerm * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)

        for j in range(len(pastLayerWeights)):
            weightGradient = errorTerm * pastLayerNodes[j]
            #pastLayerWeights[j][i] -= learningRate * weightGradient
    return weightGradient, errorTerms

def outputLayerBiasChange(lossDerivative, activationDerivative, currentLayerNodes, currentLayerBiases, trueValues, learningRate):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        lossDerivativeValue = lossDerivative(currentLayerNodes[i], trueValues[i], len(currentLayerNodes))
        if(lossDerivative == CrossEntropyLossDerivative):
            lossDerivativeValue = CrossEntropyLossDerivative(currentLayerNodes[i], trueValues[i], activationDerivative)

        if(activationDerivative == SoftMaxDerivative):
            errorTerm = lossDerivativeValue * activationDerivative(i, trueValues[i], currentLayerNodes, lossDerivative)
        else:
            errorTerm = lossDerivativeValue * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        #currentLayerBiases[i] -= learningRate * errorTerm
    return errorTerms, errorTerms

def hiddenLayerBiasChange(pastLayerErrorTerms, pastLayerWeights, currentLayerBiases, activationDerivative, currentLayerNodes, pastLayerNodes, learningRate):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        errorTerm = 0
        for node in range(len(pastLayerNodes)):
            errorTerm += pastLayerErrorTerms[node] * pastLayerWeights[node][i]
        errorTerm = errorTerm * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        #currentLayerBiases[i] -= learningRate * errorTerm
    return errorTerms, errorTerms

def backPropgation(layerNodes, weights, biases, trueValues, layers, lossFunction, learningRate):
    lossDerivatives = {
        "mse": MSEDerivative,
        "mae": MAEDerivative,
    }
    activationDerivatives = {
        main.Network.relu: ReLUDerivative,
        main.Network.sigmoid: SigmoidDerivative,
        main.Network.linear: LinearDerivative,
        main.Network.tanH: TanHDerivative,
        main.Network.softMax: SoftMaxDerivative,
    }
    
    w, weightErrorTerms = outputLayerWeightChange(lossDerivatives[lossFunction.lower()](trueValues, len(layers[len(layers) - 1])), activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], layerNodes[len(layerNodes) - 2], weights[len(weights) - 1], learningRate, trueValues)
    b, biasErrorTerms = outputLayerBiasChange(lossDerivatives[lossFunction.lower()](trueValues, len(layers[len(layers) - 1])), activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], biases[len(biases) - 1], trueValues, learningRate)
    weightGradients = [w]
    biasGradients = [b]
    for i in range(len(layers) - 2, -1, -1):
        w, weightErrorTerms = hiddenLayerWeightChange(weightErrorTerms, weights[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1], learningRate)
        b, biasErrorTerms = hiddenLayerBiasChange(biasErrorTerms, weights[i + 1], biases[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1], learningRate)
        weightGradients.insert(0, w)
        biasGradients.insert(0, b)
    return weightGradients, biasGradients

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

def ReLUDerivative(value):
    if(value > 0):
        return 1
    return 0

def SigmoidDerivative(value):
    def sigmoid(value):
        return (1 / (1 + math.exp(-value)))
    return sigmoid(value) * (1 - sigmoid(value))

def TanHDerivative(value):
    def tanH(value):
        return ((math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value)))
    return 1 - (tanH(value) ** 2)

def LinearDerivative(_):
    return 1

def SoftMaxDerivative(currValueIndex, trueValue, values, lossDerivative):
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
