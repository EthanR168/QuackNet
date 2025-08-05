from quacknet import Conv1DLayer, Conv2DLayer, PoolingLayer, GlobalAveragePooling, DenseLayer, ActivationLayer
import numpy as np

class Test_Padding:
    def test_PadImage1(self):
        inputTensor = np.ones([3, 3, 3])
        kernalSize, strideLength = 2, 2
        typeOfPadding = 0

        paddingTensor = Conv2DLayer._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

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

        paddingTensor = Conv2DLayer._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

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

        Conv = Conv2DLayer(kernalSize, 2, 1, strideLength, "no")

        Conv.kernalBiases = kernalBias
        Conv.kernalWeights = kernalWeights

        output = Conv.forward(inputTensor)

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
        usePadding = "no"

        Conv = Conv2DLayer(kernalSize, 2, 1, strideLength, usePadding)

        Conv.kernalBiases = kernalBias
        Conv.kernalWeights = kernalWeights

        output = Conv.forward(inputTensor)

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
        
        Pool = PoolingLayer(sizeOfGrid, strideLength, "max")

        output = Pool.forward(inputTensor)

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

        Pool = PoolingLayer(sizeOfGrid, strideLength, "ave")

        output = Pool.forward(inputTensor)

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

        output = GlobalAveragePooling.forward(self, inputTensor)

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

        output = DenseLayer._flatternTensor(self, inputTensor)

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

        output = ActivationLayer.forward(self, inputTensor)

        expected = np.array([
            [
                [10, -0.1, -0.01, 0],
            ]
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_Conv1D:
    def test_kernalisation1(self):
        inputTensor = np.array([
            [1, 2, 1, 2],
            [1, 1, 1, 1],
        ])
        kernalWeights = np.array([
            [
                [4, 3],
                [2, 1]
            ]
        ])

        kernalBias = [2]

        Conv = Conv1DLayer(kernalSize=2, numKernals=1, depth=2, stride=2, padding="no")

        Conv.kernalBiases = kernalBias
        Conv.kernalWeights = kernalWeights

        output = Conv.forward(inputTensor)

        expected = np.array([
            [15, 15],
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)
    
    def test_kernalisation2(self):
        inputTensor = np.array([
            [1, 2, 1, 2],
            [1, 1, 1, 1],
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

        Conv = Conv1DLayer(kernalSize=2, numKernals=2, depth=2, stride=2, padding="no")

        Conv.kernalBiases = kernalBias
        Conv.kernalWeights = kernalWeights

        output = Conv.forward(inputTensor)

        expected = np.array([
            [15, 15],
            [14, 14],
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_Conv1DPadding:
    def test_PadImage1(self):
        inputTensor = np.ones([3, 4])
        kernalSize, strideLength = 2, 2
        typeOfPadding = 0

        paddingTensor = Conv1DLayer._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

        paddingSize = int(np.ceil(((strideLength - 1) * inputTensor.shape[1] - strideLength + kernalSize) / 2))
        
        expectedPaddedTensor = []
        for layer in inputTensor:
            padded = np.concatenate([np.full(paddingSize, typeOfPadding), layer, np.full(paddingSize, typeOfPadding)])
            expectedPaddedTensor.append(padded)

        assert np.array(paddingTensor).shape == np.array(expectedPaddedTensor).shape
        assert np.allclose(paddingTensor, expectedPaddedTensor)

    def test_PadImage2(self):
        inputTensor = np.ones([2, 5])
        kernalSize, strideLength = 3, 1
        typeOfPadding = -1

        paddingTensor = Conv1DLayer._padImage(self, inputTensor, kernalSize, strideLength, typeOfPadding)

        paddingSize = int(np.ceil(((strideLength - 1) * inputTensor.shape[1] - strideLength + kernalSize) / 2))
        
        expectedPaddedTensor = []
        for layer in inputTensor:
            padded = np.concatenate([np.full(paddingSize, typeOfPadding), layer, np.full(paddingSize, typeOfPadding)])
            expectedPaddedTensor.append(padded)

        assert np.array(paddingTensor).shape == np.array(expectedPaddedTensor).shape
        assert np.allclose(paddingTensor, expectedPaddedTensor)