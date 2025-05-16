from neuralLibrary.backPropgation import outputLayerWeightChange
from neuralLibrary.lossDerivativeFunctions import CrossEntropyLossDerivative, MSEDerivative
from neuralLibrary.activationDerivativeFunctions import SoftMaxDerivative, ReLUDerivative
import numpy as np

class TestNetwork_BackPropgation_ForChangingWeights:
    def test_outputLayerWeightChange_UsingSoftmaxWithCrossEntropy(self):
        currNodes = np.array([0.8, 0.2])
        pastNodes = np.array([0.5, 0.4])
        trueValues = np.array([1, 0])

        weightGradients, errorTerms = outputLayerWeightChange(CrossEntropyLossDerivative, SoftMaxDerivative, currNodes, pastNodes, trueValues)
        
        expectedErrorTerms = currNodes - trueValues #cross entrop and softmax simplifies to precicted - true for error terms
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)

    def test_outputLayerWeightChange_ReLUWithMSE(self):
        currNodes = np.array([0.8, 0.2])
        pastNodes = np.array([0.5, 0.4])
        trueValues = np.array([1, 0])

        weightGradients, errorTerms = outputLayerWeightChange(MSEDerivative, ReLUDerivative, currNodes, pastNodes, trueValues)
        
        expectedErrorTerms = MSEDerivative(currNodes, trueValues, len(currNodes)) * ReLUDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)
    
    def test_outputLayerWeightChange_PerfectPrediction(self):
        currNodes = np.array([1, 0])
        pastNodes = np.array([0.5, 0.4])
        trueValues = np.array([1, 0])

        weightGradients, errorTerms = outputLayerWeightChange(MSEDerivative, ReLUDerivative, currNodes, pastNodes, trueValues)
        
        expectedErrorTerms = MSEDerivative(currNodes, trueValues, len(currNodes)) * ReLUDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)