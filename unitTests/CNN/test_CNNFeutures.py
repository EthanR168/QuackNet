from quacknet.CNN.convulationalFeutures import ConvulationalNetwork
import numpy as np

class Test_Padding:
    def test_PadImage1(self):
        inputTensor = np.ones([3, 3, 3])
        kernalSize, strideLength = 2, 2
        typeOfPadding = 0

        paddingTensor = ConvulationalNetwork._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

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

        paddingTensor = ConvulationalNetwork._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

        paddingSize = int(np.ceil(((strideLength - 1) * inputTensor.shape[1] - strideLength + kernalSize) / 2))
        expectedPaddedTensor = []
        for layer in inputTensor:
            expectedPaddedTensor.append(np.pad(layer, (paddingSize,paddingSize)))
            
        assert np.array(paddingTensor).shape == np.array(expectedPaddedTensor).shape
        assert np.allclose(paddingTensor, expectedPaddedTensor)

class Test_kernalisation:
    def test_kernalisation1(self):
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
        [
            [4, 3],
            [2, 1]
        ]
        ])
        kernalBias = [2]
        kernalSize = strideLength = 2
        usePadding = False

        output = ConvulationalNetwork._kernalisation(self, inputTensor, kernalWeights, kernalBias, kernalSize, strideLength=strideLength, usePadding=usePadding)

        '''
        [1, 2]      X       [4, 3]    =   [4, 6]     =  20
        [3, 4]              [2, 1]        [6, 4]

        [1, 2]      X       [4, 3]    =   [4, 6]     =  20
        [3, 4]              [2, 1]        [6, 4]

        [20, 20]
        [20, 20]

        [1, 1]      X       [4, 3]    =   [4, 3]     =  10
        [1, 1]              [2, 1]        [2, 1]

        [1, 1]      X       [4, 3]    =   [4, 3]     =  10
        [1, 1]              [2, 1]        [2, 1]

        [10, 10]    +     [20, 20]     +     2    =    [32, 32]
        [10, 10]    +     [20, 20]                     [32, 32]
        '''

        expected = np.array([
        [
            [32, 32],
            [32, 32],
        ]
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)
    
    def test_kernalisation2(self):
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
        [
            [4, 3],
            [2, 1]
        ],
        [
            [2, 2],
            [2, 2]
        ]
        ])
        kernalBias = [2, 4]
        kernalSize = strideLength = 2
        usePadding = False

        output = ConvulationalNetwork._kernalisation(self, inputTensor, kernalWeights, kernalBias, kernalSize, strideLength=strideLength, usePadding=usePadding)

        '''
        Kernal 1:
        [1, 2]      X       [4, 3]    =   [4, 6]     =  20
        [3, 4]              [2, 1]        [6, 4]

        [1, 2]      X       [4, 3]    =   [4, 6]     =  20
        [3, 4]              [2, 1]        [6, 4]

        [20, 20]
        [20, 20]

        [1, 1]      X       [4, 3]    =   [4, 3]     =  10
        [1, 1]              [2, 1]        [2, 1]

        [1, 1]      X       [4, 3]    =   [4, 3]     =  10
        [1, 1]              [2, 1]        [2, 1]

        [10, 10]    +     [20, 20]     +     2    =    [32, 32]
        [10, 10]    +     [20, 20]                     [32, 32]

        
        Kernal 2:
        [1, 2]      X       [2, 2]    =   [2, 4]     =  20
        [3, 4]              [2, 2]        [6, 8]

        [1, 2]      X       [2, 2]    =   [2, 4]     =  20
        [3, 4]              [2, 2]        [6, 8]

        [20, 20]
        [20, 20]

        [1, 1]      X       [2, 2]    =   [2, 2]     =  8
        [1, 1]              [2, 2]        [2, 2]

        [1, 1]      X       [2, 2]    =   [2, 2]     =  8
        [1, 1]              [2, 2]        [2, 2]

        [8, 8]    +     [20, 20]     +     4    =    [32, 32]
        [8, 8]    +     [20, 20]                     [32, 32]
        '''

        expected = np.array([
        [
            [32, 32],
            [32, 32],
        ],
        [
            [32, 32],
            [32, 32],
        ]
        ])
        
        print(expected)
        print(output)

        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_Pooling:
    def test_maxPooling(self):
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
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        ]) 
        sizeOfGrid = strideLength = 2

        output = ConvulationalNetwork._pooling(self, inputTensor, sizeOfGrid, strideLength, "max")

        expected = np.array([
        [
            [4, 4],
            [4, 4],
        ],
        [
            [1, 1],
            [1, 1],
        ],
        [
            [6, 8],
            [14, 16],
        ]
        ])

        assert expected.shape == output.shape
        assert np.allclose(expected, output)

    def test_averagePooling2(self):
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
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        ]) 
        sizeOfGrid = strideLength = 2

        output = ConvulationalNetwork._pooling(self, inputTensor, sizeOfGrid, strideLength, "ave")

        expected = np.array([
        [
            [2.5, 2.5],
            [2.5, 2.5],
        ],
        [
            [1, 1],
            [1, 1],
        ],
        [
            [3.5, 5.5],
            [11.5, 13.5],
        ]
        ])
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_PoolingGlobalAverage:
    def test_poolingGlobalAverage(self):
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
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        ]) 

        output = ConvulationalNetwork._poolingGlobalAverage(self, inputTensor)

        expected = np.array([2.5, 1, 8.5])

        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_Flattening:
    def test_flattening(self):
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
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
        ]) 

        output = ConvulationalNetwork._flatternTensor(self, inputTensor)

        expected = np.array([1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_ActivationLayer:
    def test_activationLayer(self):
        inputTensor = np.array([
            [
                [10, -10, -1, 0],
            ]
        ]) 

        output = ConvulationalNetwork._activation(self, inputTensor)

        expected = np.array([
            [
                [10, -0.1, -0.01, 0],
            ]
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

