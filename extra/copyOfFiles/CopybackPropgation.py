from activationFunctions import relu, sigmoid, tanH, linear, softMax
from activationDerivativeFunctions import ReLUDerivative, SigmoidDerivative, TanHDerivative, LinearDerivative, SoftMaxDerivative
from lossDerivativeFunctions import MSEDerivative, MAEDerivative, CrossEntropyLossDerivative
from lossFunctions import MSELossFunction, MAELossFunction, CrossEntropyLossFunction

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

def outputLayerWeightChange(lossDerivative, activationDerivative, currentLayerNodes, pastLayerNodes, trueValues):
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
        w = []
        for j in range(len(pastLayerNodes)):
            w.append(errorTerm * pastLayerNodes[j])
        weightGradients.append(w)
        errorTerms.append(errorTerm)
    return weightGradients, errorTerms

def hiddenLayerWeightChange(pastLayerErrorTerms, pastLayerWeights, activationDerivative, currentLayerNodes, pastLayerNodes):
    errorTerms = []
    weightGradients = []
    for i in range(len(currentLayerNodes)):
        errorTerm = 0
        for node in range(len(pastLayerNodes)):
            errorTerm += pastLayerErrorTerms[node] * pastLayerWeights[i][node]
        errorTerm = errorTerm * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        w = []
        for j in range(len(pastLayerNodes)):
            w.append(errorTerm * pastLayerNodes[j])
        weightGradients.append(w)
    return weightGradients, errorTerms

def outputLayerBiasChange(lossDerivative, activationDerivative, currentLayerNodes, trueValues):
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

def hiddenLayerBiasChange(pastLayerErrorTerms, pastLayerWeights, activationDerivative, currentLayerNodes, pastLayerNodes):
    errorTerms = []
    for i in range(len(currentLayerNodes)):
        errorTerm = 0
        for node in range(len(pastLayerNodes)):
            errorTerm += pastLayerErrorTerms[node] * pastLayerWeights[i][node]
        errorTerm = errorTerm * activationDerivative(currentLayerNodes[i])
        errorTerms.append(errorTerm)
        #currentLayerBiases[i] -= learningRate * errorTerm
    return errorTerms, errorTerms

def backPropgation(layerNodes, weights, biases, trueValues, layers, lossFunction):
    lossDerivatives = {
        MSELossFunction: MSEDerivative,
        MAELossFunction: MAEDerivative,
    }
    activationDerivatives = {
        relu: ReLUDerivative,
        sigmoid: SigmoidDerivative,
        linear: LinearDerivative,
        tanH: TanHDerivative,
        softMax: SoftMaxDerivative,
    }
    #lossDerivative = lossDerivatives[lossFunction](layerNodes[len(layers) - 1][0], trueValues, layers[len(layers) - 1][0])
    w, weightErrorTerms = outputLayerWeightChange(lossDerivatives[lossFunction], activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], layerNodes[len(layerNodes) - 2], trueValues)
    b, biasErrorTerms = outputLayerBiasChange(lossDerivatives[lossFunction], activationDerivatives[layers[len(layers) - 1][1]], layerNodes[len(layerNodes) - 1], trueValues)
    weightGradients = [w]
    biasGradients = [b]
    for i in range(len(layers) - 2, -1, -1):
        w, weightErrorTerms = hiddenLayerWeightChange(weightErrorTerms, weights[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1])
        b, biasErrorTerms = hiddenLayerBiasChange(biasErrorTerms, weights[i], activationDerivatives[layers[i][1]], layerNodes[i], layerNodes[i + 1])
        weightGradients.insert(0, w)
        biasGradients.insert(0, b)
    return weightGradients, biasGradients
