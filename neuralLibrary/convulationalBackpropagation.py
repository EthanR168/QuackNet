import numpy as np

class CNNbackpropagation:
    def ConvolutionDerivative(self, errorPatch, kernals, inputTensor, stride):
        '''
        gets the error gradient from the layer infront and it is a error patch
        this error patch is the same size as what the convolutional layer outputed during forward propgation
        get the kernal (as in a patch of the image) again, but this time you are multipling each value in the kernal by 1 value that is inside the error patch
        this makes the gradient of the loss of one kernal's weight
        
        the gradient of the loss of one kernal's bias is the summ of all the error terms
        because bias is applied to every input in forward propgation
        
        the gradient of the loss of the input, which is the error terms for the layer behind it
        firstly the kernal has to be flipped, meaning flip the kernal left to right and then top to bottom, but not flipping the layers,
        the gradient of one pixel, is the summ of each error term multiplied by the flipped kernal 
        '''

        kernalSize = len(kernals[0]) #all kernals are the same shape and squares
        weightGradients = np.zeros_like(kernals) #kernals are the same size
        outputHeight = len(inputTensor[0]) - kernalSize + 1
        outputWidth = len(inputTensor[0][0]) - kernalSize + 1
        for output in range(len(kernals)):
            for layer in range(len(inputTensor)):
                for i in range(0, outputHeight, stride):
                    for j in range(0, outputWidth, stride):
                        kernal = inputTensor[layer][i: i + kernalSize, j: j + kernalSize]
                        kernal = kernal * errorPatch[output][i // stride][j // stride]
                        weightGradients[:, :, layer, output] += kernal
        
        biasGradients = np.zeros(len(kernals))
        for output in range(len(kernals)):
            biasGradients[output] = np.sum(errorPatch[output])

        inputErrorTerms = np.zeros_like(inputTensor)
        kernalSize = len(kernals[0]) 
        for output in range(len(errorPatch)):
            for layer in range(len(inputTensor)):
                flipped = kernal[output, layer, ::-1, ::-1]
                for i in range(0, outputHeight, stride):
                    for j in range(0, outputWidth, stride):
                        errorKernal = errorPatch[output, i, j]
                        inputErrorTerms[layer, i: i + kernalSize, j: j + kernalSize] += errorKernal * flipped
        
        return weightGradients, biasGradients, inputErrorTerms
            