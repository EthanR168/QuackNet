from neuralLibrary.backPropgation import outputLayerWeightChange, hiddenLayerWeightChange, outputLayerBiasChange, hiddenLayerBiasChange
from neuralLibrary.lossDerivativeFunctions import CrossEntropyLossDerivative, MSEDerivative
from neuralLibrary.activationDerivativeFunctions import SoftMaxDerivative, ReLUDerivative, SigmoidDerivative, TanHDerivative, LinearDerivative
import numpy as np

class TestNetwork_BackPropgation_Weights_Output:
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

class TestNetwork_BackPropgation_Weights_Hidden:
    def test_hiddenLayerWeightChange_ReLU(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        weightGradients, errorTerms = hiddenLayerWeightChange(errorTermsNextLayer, pastWeights, ReLUDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * ReLUDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)

    def test_hiddenLayerWeightChange_Sigmoid(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        weightGradients, errorTerms = hiddenLayerWeightChange(errorTermsNextLayer, pastWeights, SigmoidDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * SigmoidDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)

    def test_hiddenLayerWeightChange_TanH(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        weightGradients, errorTerms = hiddenLayerWeightChange(errorTermsNextLayer, pastWeights, TanHDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * TanHDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)

    def test_hiddenLayerWeightChange_Linear(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        weightGradients, errorTerms = hiddenLayerWeightChange(errorTermsNextLayer, pastWeights, LinearDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * LinearDerivative(currNodes)
        expectedWeightGradients = np.outer(pastNodes, expectedErrorTerms)

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedWeightGradients)

class TestNetwork_BackPropgation_Biases_Output:
    def test_outputLayerBiasesChange_UsingSoftmaxWithCrossEntropy(self):
        currNodes = np.array([0.8, 0.2])
        trueValues = np.array([1, 0])

        weightGradients, errorTerms = outputLayerBiasChange(CrossEntropyLossDerivative, SoftMaxDerivative, currNodes, trueValues)
        
        expectedErrorTerms = currNodes - trueValues #cross entrop and softmax simplifies to precicted - true for error terms
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(weightGradients, expectedBiasGradients)

    def test_outputLayerBiasesChange_ReLUWithMSE(self):
        currNodes = np.array([0.8, 0.2])
        trueValues = np.array([1, 0])

        biasGradients, errorTerms = outputLayerBiasChange(MSEDerivative, ReLUDerivative, currNodes, trueValues)
        
        expectedErrorTerms = MSEDerivative(currNodes, trueValues, len(currNodes)) * ReLUDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)
    
    def test_outputLayerBiasesChange_PerfectPrediction(self):
        currNodes = np.array([1, 0])
        trueValues = np.array([1, 0])

        biasGradients, errorTerms = outputLayerBiasChange(MSEDerivative, ReLUDerivative, currNodes, trueValues)
        
        expectedErrorTerms = MSEDerivative(currNodes, trueValues, len(currNodes)) * ReLUDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)

class TestNetwork_BackPropgation_Biases_Hidden:
    def test_hiddenLayerBiasChange_ReLU(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        biasGradients, errorTerms = hiddenLayerBiasChange(errorTermsNextLayer, pastWeights, ReLUDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * ReLUDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)

    def test_hiddenLayerBiasChange_Sigmoid(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        biasGradients, errorTerms = hiddenLayerBiasChange(errorTermsNextLayer, pastWeights, SigmoidDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * SigmoidDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)

    def test_hiddenLayerBiasChange_TanH(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        biasGradients, errorTerms = hiddenLayerBiasChange(errorTermsNextLayer, pastWeights, TanHDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * TanHDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)

    def test_hiddenLayerBiasChange_Linear(self):
        errorTermsNextLayer = np.array([0.1 , -0.2])
        currNodes = np.array([0.8, 0.0])
        pastNodes = np.array([0.5, 0.4])
        pastWeights = np.array([[0.4, 0.3], [0.2, 0.1]])

        biasGradients, errorTerms = hiddenLayerBiasChange(errorTermsNextLayer, pastWeights, LinearDerivative, currNodes, pastNodes)
    
        expectedErrorTerms = errorTermsNextLayer @ pastWeights.T
        expectedErrorTerms = expectedErrorTerms * LinearDerivative(currNodes)
        expectedBiasGradients = expectedErrorTerms

        assert np.allclose(errorTerms, expectedErrorTerms)
        assert np.allclose(biasGradients, expectedBiasGradients)
