import math

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
    for i in range(len(currentLayerNodes)):
        errorTerm = lossDerivative(currentLayerNodes[i], trueValues[i]) * activationDerivative(currentLayerNodes[i])
        for j in range(len(pastLayerNodes)):
            weightGradient = errorTerm * pastLayerNodes[j]
            pastLayerWeights[j][i] -= learningRate * weightGradient 
        errorTerms.append(errorTerm)
    return pastLayerWeights, errorTerms

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
            pastLayerWeights[j][i] -= learningRate * weightGradient
    return pastLayerWeights, errorTerms

def outputLayerBiaseChange(lossDerivative, activationDerivative, currentLayerNodes, currentLayerBiases, trueValues, learningRate):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        errorTerm = lossDerivative(currentLayerNodes[i], trueValues[i]) * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        currentLayerBiases[i] -= learningRate * errorTerm
    return currentLayerBiases, errorTerms

def hiddenLayerBiaseChange(pastLayerErrorTerms, pastLayerWeights, currentLayerBiases, activationDerivative, currentLayerNodes, pastLayerNodes, learningRate):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        errorTerm = 0
        for node in range(len(pastLayerNodes)):
            errorTerm += pastLayerErrorTerms[node] * pastLayerWeights[node][i]
        errorTerm = errorTerm * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        currentLayerBiases[i] -= learningRate * errorTerm
    return currentLayerBiases, errorTerms

def backPropagation(layerNodes, weights, biases, trueValues, layers, lossFunction, learningRate):
    lossDerivatives = {
        "mse": MSEDerivative,
    }
    activationDerivatives = {
        "relu": ReLUDerivative,
        "sigmoid": SigmoidDerivative,
        "linear": LinearDerivative,
    }
    weights[len(weights) - 1], weightErrorTerms = outputLayerWeightChange(lossDerivatives[lossFunction.lower()], activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], layerNodes[len(layerNodes) - 2], weights[len(weights) - 1], learningRate, trueValues)
    biases[len(biases) - 1], biasErrorTerms = outputLayerBiaseChange(lossDerivatives[lossFunction.lower()], activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], biases[len(biases) - 1], trueValues, learningRate)
    for i in range(len(layers) - 2, -1, -1):
        weights[i], weightErrorTerms = hiddenLayerWeightChange(weightErrorTerms, weights[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1], learningRate)
        biases[i], biasErrorTerms = hiddenLayerBiaseChange(biasErrorTerms, weights[i + 1], biases[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1], learningRate)

def MSEDerivative(value, trueValue, sizeOfLayer):
    return 2 * (trueValue - value) / sizeOfLayer

def ReLUDerivative(value):
    if(value > 0):
        return 1
    return 0

def SigmoidDerivative(value):
    def sigmoid(value):
        return (1 / (1 + math.exp(-value)))
    return sigmoid(value) * (1 - sigmoid(value))

def LinearDerivative(_):
    return 1
