import numpy as np
import math

'''
Covulutional neural network:
Kernels:
-is a cube (eg. 2 x 2 x 2) that cotains weights
-it moves across the image starting from the left to the right, with each step being a stride (eg. 2)
-once the cube reaches the very right it goes to the row underneath and starts at the very left pixel of that row (so no bouncing back and forth)
-the cube calculates the dot product of its weight and the pixel values in the image under the cube for all layers
-when the cube is moving it may not be over the image so you add padding
-the padding can be either 1 or 0, or you can ignore thos values which means the image will get smaller

Activation:
-goes through all the data that the last layer made and uses either relu or leaky relu

Pooling:
-is a grid (eg. 2 x 2) that moves the same as kernels
-however the grid can either be max pooling or average pooling
-max pooling gets the highest value in the grid of the image whilst average gets the average

Neural Network:
-flattens the tensors and inputs into a neural network
'''

class ConvulationalNetwork:
    def padImage(self, image, kernalSize, strideLength, typeOfPadding): #pads image
        paddingSize = math.ceil(((strideLength - 1) * len(image) - strideLength + kernalSize) / 2)
        padding = np.full((image.shape[0] + paddingSize * 2, image.shape[1] + paddingSize * 2), typeOfPadding) #creates an 2d numpy array of size paddingSize x paddingSize
        padding[paddingSize: paddingSize + image.shape[0], paddingSize: paddingSize + image.shape[1]] = image
        return padding

    def kernalisation(self, inputTensor, kernals, kernalsBiases, sizeOfGrid = 2, usePadding = True, typeOfPadding= 0, strideLength = 1):
        tensorKernals = []
        if(usePadding == True):
            image = self.padImage(inputTensor, sizeOfGrid, strideLength, typeOfPadding)
        else:
            image = inputTensor
        outputHeight = (image.shape[0] - sizeOfGrid) // strideLength + 1
        outputWidth = (image.shape[1] - sizeOfGrid) // strideLength + 1
        for i in range(len(kernals)):
            output = np.zeros((outputHeight, outputWidth))
            kernal = kernals[i]
            biases = kernalsBiases[i]
            for x in range(outputHeight):
                for y in range(outputWidth):
                    indexX = x * strideLength
                    indexY = y * strideLength
                    gridOfValues = image[indexX: indexX + sizeOfGrid, indexY: indexY + sizeOfGrid, :] # 3d tensor
                    dotProduct = np.sum(gridOfValues * kernal) + biases
                    output[x, y] = dotProduct
            tensorKernals.append(output)
        return np.stack(tensorKernals, axis = -1) #tensorKernals = (outputHeight, outputWidth, numberOfKernals)
                    
    def activation(self, inputTensor):
        alpha = 0.01
        return np.maximum(inputTensor, inputTensor * alpha)

    def pooling(self, inputTensor, sizeOfGrid = 2, strideLength = 1, typeOfPooling = "max"):
        tensorPools = []
        for image in inputTensor: # tensor is a 3d structures, so it is turning it into a 2d array (eg. an layer or image)
            outputHeight = (image.shape[0] - sizeOfGrid) // strideLength + 1
            outputWidth = (image.shape[1] - sizeOfGrid) // strideLength + 1
            output = np.zeros((outputHeight, outputWidth))
            for x in range(outputHeight):
                for y in range(outputWidth):
                    indexX = x * strideLength
                    indexY = y * strideLength
                    gridOfValues = image[indexX: indexX + sizeOfGrid, indexY: indexY + sizeOfGrid]
                    if(typeOfPooling == "max"):
                        value = np.max(gridOfValues)
                    else:
                        value = np.mean(gridOfValues)
                    output[x, y] = value
            tensorPools.append(output)
        return tensorPools

    def flatternTensor(self, inputTensor):
        return inputTensor.reshape(-1)

