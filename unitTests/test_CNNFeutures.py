from neuralLibrary.convulationalFeutures import ConvulationalNetwork
import numpy as np

class Test_Padding:
    def test_PadImage1(self):
        inputTensor = np.ones([3, 3, 3])
        kernalSize, strideLength = 2, 2
        typeOfPadding = 0

        paddingTensor = ConvulationalNetwork.padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

        paddingSize = int(np.ceil(((strideLength - 1) * inputTensor.shape[1] - strideLength + kernalSize) / 2))
        expectedPaddedTensor = []
        for layer in inputTensor:
            expectedPaddedTensor.append(np.pad(layer, (paddingSize,paddingSize)))

        assert np.array(paddingTensor).shape == np.array(expectedPaddedTensor).shape
        assert np.allclose(paddingTensor, expectedPaddedTensor)

    def test_PadImage2(self):
        inputTensor = np.ones([5, 6, 6])
        kernalSize, strideLength = 2, 2
        typeOfPadding = 0

        paddingTensor = ConvulationalNetwork.padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

        paddingSize = int(np.ceil(((strideLength - 1) * inputTensor.shape[1] - strideLength + kernalSize) / 2))
        expectedPaddedTensor = []
        for layer in inputTensor:
            expectedPaddedTensor.append(np.pad(layer, (paddingSize,paddingSize)))
            
        assert np.array(paddingTensor).shape == np.array(expectedPaddedTensor).shape
        assert np.allclose(paddingTensor, expectedPaddedTensor)

class Test_kernalisation:
    def test_kernalisation(self):
        inputTensor = np.array([
            [
                [1, 2, 1, 2],
                [3, 4, 3, 4],
                [1, 2, 1, 2],
                [3, 4, 3, 4],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ])
        kernalWeights = np.array([
            [4, 3],
            [2, 1]
        ])
        kernalBias = [2]
        kernalSize = strideLength = 2
        usePadding = False

        output = ConvulationalNetwork.kernalisation(self, inputTensor, kernalWeights, kernalBias, kernalSize, strideLength=strideLength, usePadding=usePadding)

        '''
        [1, 2]      X       [4, 3]     =   [6, 8]     +   2   =   30
        [3, 4]              [2, 1]         [8, 6]

        [3, 4]      X       [4, 3]     =   [14, 14]   +   2   =  38
        [1, 2]              [2, 1]         [4, 4]

        [1, 2]      X       [4, 3]     =   [6, 8]     +   2   =  30
        [3, 4]              [2, 1]         [8, 6]

        [30, 30]
        [38, 38]
        [30, 30]
        '''

        expected = np.array(
            [30, 30],
            [38, 38],
            [30, 30],
        )

        assert expected.shape == output.shape
        assert expected == output