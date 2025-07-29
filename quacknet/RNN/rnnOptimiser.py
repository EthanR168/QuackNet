import numpy as np

class RNNOptimiser():
    def _AdamsOptimiserWithBatches(self, inputData, labels, inputWeights, hiddenWeights, biases, outputWeights, outputBiases, batchSize, alpha, beta1, beta2, epsilon):
        """
        Performs Adam optimisation on the CNN weights and biases using mini batches.

        Args:
            inputData (ndarray): All the training data.
            labels (ndarray): All the true labels for the training data.
            weights (list of ndarray): Current weights of the CNN layers.
            biases (list of ndarray): Current biases of the CNN layers.
            batchSize (int): Size of batches.
            alpha (float): Learning rate.
            beta1 (float): Decay rate for the first moment.
            beta2 (float): Decay rate for the second moment. 
            epsilon (float): Small constant to avoid division by zero.
        
        Returns: 
            allNodes (list): List of layers for each input processed.
            weights (list of ndarray): Updated weights after optimisation.
            biases (list of ndarray): Updated biases after optimisation.
        """
        
        Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, Output_weightGradients, Output_biasGradients, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, Input_weightGradients, Biases_biasGradients, Hidden_firstMomentWeight, Hidden_secondMomentWeight, Hidden_weightGradients = self._InitialiseEverything(inputWeights, hiddenWeights, biases, outputWeights, outputBiases)
        allNodes = []
        for i in range(0, len(inputData), batchSize):
            batchData = inputData[i:i+batchSize]
            batchLabels = labels[i:i+batchSize]
            for j in range(len(batchData)):
                layerNodes = self.RNNForwardPropagation(batchData[j])
                allNodes.append(layerNodes)

                inputWeightGrad, hiddenStateWeightGrad, biasGrad, outputWeightGrad, outputbiasGrad = self.RNNBackwardPropagation(layerNodes, labels[i])
                
                Output_weightGradients, Output_biasGradients = self._addGradients(1, Output_weightGradients, Output_biasGradients, outputWeightGrad, outputbiasGrad)
                Input_weightGradients, Biases_biasGradients = self._addGradients(1, Input_weightGradients, Biases_biasGradients, inputWeightGrad, biasGrad)
                Hidden_weightGradients, _ = self._addGradients(1, Hidden_weightGradients, Biases_biasGradients, hiddenStateWeightGrad, biasGrad)
            
            outputWeights, outputBiases, Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias = self._Adams(Output_weightGradients, Output_biasGradients, outputWeights, outputBiases, i + 1, Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, alpha, beta1, beta2, epsilon)
            inputWeights, biases, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias = self._Adams(Input_weightGradients, Biases_biasGradients, inputWeights, biases, i + 1, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, alpha, beta1, beta2, epsilon)
            hiddenWeights, _, Hidden_firstMomentWeight, _, Hidden_secondMomentWeight, _ = self._Adams(Hidden_weightGradients, Biases_biasGradients, hiddenStateWeightGrad, outputBiases, i + 1, Hidden_firstMomentWeight, Output_firstMomentBias, Hidden_secondMomentWeight, Output_secondMomentBias, alpha, beta1, beta2, epsilon)
            
            Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, Output_weightGradients, Output_biasGradients, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, Input_weightGradients, Biases_biasGradients, Hidden_firstMomentWeight, Hidden_secondMomentWeight, Hidden_weightGradients = self._InitialiseEverything(inputWeights, hiddenWeights, biases, outputWeights, outputBiases)
        
        return allNodes, inputWeights, hiddenWeights, biases, outputWeights, outputBiases

    def _AdamsOptimiserWithoutBatches(self, inputData, labels, inputWeights, hiddenWeights, biases, outputWeights, outputBiases, alpha, beta1, beta2, epsilon):
        """
        Performs Adam optimisation on the CNN weights and biases without using batches.

        Args:
            inputData (ndarray): All the training data.
            labels (ndarray): All the true labels for the training data.
            weights (list of ndarray): Current weights of the CNN layers.
            biases (list of ndarray): Current biases of the CNN layers.
            alpha (float): Learning rate.
            beta1 (float): Decay rate for the first moment.
            beta2 (float): Decay rate for the second moment. 
            epsilon (float): Small constant to avoid division by zero.
        
        Returns: 
            allNodes (list): List of layers for each input processed.
            weights (list of ndarray): Updated weights after optimisation.
            biases (list of ndarray): Updated biases after optimisation.
        """
        
        Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, Output_weightGradients, Output_biasGradients, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, Input_weightGradients, Biases_biasGradients, Hidden_firstMomentWeight, Hidden_secondMomentWeight, Hidden_weightGradients = self._InitialiseEverything(inputWeights, hiddenWeights, biases, outputWeights, outputBiases)
        allNodes = []
        for i in range(len(inputData)):
            layerNodes = self.RNNForwardPropagation(inputData[i])
            allNodes.append(layerNodes)

            inputWeightGrad, hiddenStateWeightGrad, biasGrad, outputWeightGrad, outputbiasGrad = self.RNNBackwardPropagation(layerNodes, labels[i])
            
            Output_weightGradients, Output_biasGradients = self._addGradients(1, Output_weightGradients, Output_biasGradients, outputWeightGrad, outputbiasGrad)
            Input_weightGradients, Biases_biasGradients = self._addGradients(1, Input_weightGradients, Biases_biasGradients, inputWeightGrad, biasGrad)
            Hidden_weightGradients, _ = self._addGradients(1, Hidden_weightGradients, Biases_biasGradients, hiddenStateWeightGrad, biasGrad)
            
            outputWeights, outputBiases, Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias = self._Adams(Output_weightGradients, Output_biasGradients, outputWeights, outputBiases, i + 1, Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, alpha, beta1, beta2, epsilon)
            inputWeights, biases, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias = self._Adams(Input_weightGradients, Biases_biasGradients, inputWeights, biases, i + 1, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, alpha, beta1, beta2, epsilon)
            hiddenWeights, _, Hidden_firstMomentWeight, _, Hidden_secondMomentWeight, _ = self._Adams(Hidden_weightGradients, Biases_biasGradients, hiddenStateWeightGrad, outputBiases, i + 1, Hidden_firstMomentWeight, Output_firstMomentBias, Hidden_secondMomentWeight, Output_secondMomentBias, alpha, beta1, beta2, epsilon)
            
            Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, Output_weightGradients, Output_biasGradients, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, Input_weightGradients, Biases_biasGradients, Hidden_firstMomentWeight, Hidden_secondMomentWeight, Hidden_weightGradients = self._InitialiseEverything(inputWeights, hiddenWeights, biases, outputWeights, outputBiases)
        return allNodes, inputWeights, hiddenWeights, biases, outputWeights, outputBiases

    def _Adams(self, weightGradients, biasGradients, weights, biases, timeStamp, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias, alpha, beta1, beta2, epsilon):
        """
        Performs a single Adam optimisation update on weights and biases.

        Args:
            weightGradients (list of ndarray): Gradients of the weights.
            biasGradients (list of ndarray): Gradients of the biases.
            weights (list of ndarray): Current weights.
            biases (list of ndarray): Current biases.
            timeStamp (int): The current time step, used for bias correction.
            firstMomentWeight (list of ndarray): First moment estimates for weights.
            firstMomentBias (list of ndarray): First moment estimates for biases.
            secondMomentWeight (list of ndarray): Second moment estimates for weights.
            secondMomentBias (list of ndarray): Second moment estimates for biases.
            alpha (float): Learning rate.
            beta1 (float): Decay rate for the first moment.
            beta2 (float): Decay rate for the second moment. 
            epsilon (float): Small constant to avoid division by zero.
        
        Returns: 
            weights (list of ndarray): Updated weights after optimisation.
            biases (list of ndarray): Updated biases after optimisation.
            firstMomentWeight (list of ndarray): Updated firstMomentWeight after optimisation.
            firstMomentBias (list of ndarray): Updated firstMomentBias after optimisation.
            secondMomentWeight (list of ndarray): Updated secondMomentWeight after optimisation.
            secondMomentBias (list of ndarray): Updated secondMomentBias after optimisation.
        """
        for i in range(len(weights)):
            firstMomentWeight[i] = beta1 * np.array(firstMomentWeight[i]) + (1 - beta1) * weightGradients[i]
            secondMomentWeight[i] = beta2 * np.array(secondMomentWeight[i]) + (1 - beta2) * (weightGradients[i] ** 2)

            firstMomentWeightHat = firstMomentWeight[i] / (1 - beta1 ** timeStamp)
            secondMomentWeightHat = secondMomentWeight[i] / (1 - beta2 ** timeStamp)

            weights[i] -= alpha * firstMomentWeightHat / (np.sqrt(secondMomentWeightHat) + epsilon)
        
        for i in range(len(biases)):
            firstMomentBias[i] = beta1 * np.array(firstMomentBias[i]) + (1 - beta1) * np.array(biasGradients[i])
            secondMomentBias[i] = beta2 * np.array(secondMomentBias[i]) + (1 - beta2) * (np.array(biasGradients[i]) ** 2)

            firstMomentBiasHat = firstMomentBias[i] / (1 - beta1 ** timeStamp)
            secondMomentBiasHat = secondMomentBias[i] / (1 - beta2 ** timeStamp)

            biases[i] -= alpha * firstMomentBiasHat / (np.sqrt(secondMomentBiasHat) + epsilon)
        return weights, biases, firstMomentWeight, firstMomentBias, secondMomentWeight, secondMomentBias

    def _initialiseGradients(self, weights, biases):
        """
        Initialise the weight and bias gradients as zero arrays with the same shape as weights and biases.

        Args:
            weights (list of ndarray): The weights of the CNN layers.
            biases (list of ndarray): The biases of the CNN layers.

        Returns:
            weightGradients (list of ndarray): Initialised gradients for weights.
            biasGradients (list of ndarray): Initialised gradients for biases.
        """
        weightGradients, biasGradients = [], []
        for i in weights:
            w = []
            for j in i:
                w.append(np.zeros_like(j, dtype=np.float64))
            weightGradients.append(w)
        for i in biases:
            b = []
            for j in i:
                b.append(np.zeros_like(j, dtype=np.float64))
            biasGradients.append(b)
        return weightGradients, biasGradients

    def _addGradients(self, batchSize, weightGradients, biasGradients, w, b):
        """
        Adds gardients from a batch to the accumulated gradients.

        Args:
            batchSize (int): Number of samples in the current batch.
            weightGradients (list of ndarray): Accumulated weight gradients.
            biasGradients (list of ndarray): Accumulated bias gradients. 
            w (list of ndarray): Gradients of the weights from the current batch.
            b (list of ndarray): Gradients of the biases from the current batch.
        
        Returns:
            weightGradients (list of ndarray): Updated accumulated weight gradients.
            biasGradients (list of ndarray): Updated accumulated bias gradients. 
        """
        for i in range(len(weightGradients)):
            weightGradients[i] += np.array(w[i]) / batchSize
            #weightGradients[i] = np.clip(weightGradients[i], -1, 1)

        for i in range(len(biasGradients)):
            biasGradients[i] += np.array(b[i]) / batchSize
            #biasGradients[i] = np.clip(biasGradients[i], -1, 1)
        return weightGradients, biasGradients

    def _initialiseMoment(self, weights, biases):
        """
        Initialise the first and second moment estimates for Adam optimiser as zero arrays matching weights and biases.

        Args:
            weights (list of ndarray): The weights of the CNN layers.
            biases (list of ndarray): The biases of the CNN layers.

        Returns:
            momentWeight (list of ndarray): Initialised moments for weights.
            momentBias (list of ndarray): Initialised moments for biases.
        """
        momentWeight = []
        momentBias = []
        for i in weights:
            w = []
            for j in i:
                w.append(np.zeros_like(j))
            momentWeight.append(w)
        for i in biases:
            b = []
            for j in i:
                b.append(np.zeros_like(j))
            momentBias.append(b)
        return momentWeight, momentBias

    def _InitialiseEverything(self, inputWeights, hiddenWeights, biases, outputWeights, outputBiases):
        Output_firstMomentWeight, Output_firstMomentBias = self._initialiseMoment(outputWeights, outputBiases)
        Output_secondMomentWeight, Output_secondMomentBias = self._initialiseMoment(outputWeights, outputBiases)
        Output_weightGradients, Output_biasGradients = self._initialiseGradients(outputWeights, outputBiases)

        Input_firstMomentWeight, Biases_firstMomentBias = self._initialiseMoment(inputWeights, biases)
        Input_secondMomentWeight, Biases_secondMomentBias = self._initialiseMoment(inputWeights, biases)
        Input_weightGradients, Biases_biasGradients = self._initialiseGradients(inputWeights, biases)

        Hidden_firstMomentWeight, _ = self._initialiseMoment(hiddenWeights, outputBiases)
        Hidden_secondMomentWeight, _ = self._initialiseMoment(hiddenWeights, outputBiases)
        Hidden_weightGradients, _ = self._initialiseGradients(hiddenWeights, outputBiases)
        return Output_firstMomentWeight, Output_firstMomentBias, Output_secondMomentWeight, Output_secondMomentBias, Output_weightGradients, Output_biasGradients, Input_firstMomentWeight, Biases_firstMomentBias, Input_secondMomentWeight, Biases_secondMomentBias, Input_weightGradients, Biases_biasGradients, Hidden_firstMomentWeight, Hidden_secondMomentWeight, Hidden_weightGradients
    
