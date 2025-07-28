from quacknet.CNN.convulationalManager import CNNModel, ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from quacknet.main import Network
import numpy as np

class Test_ConvLayer:
    def test_ConvLayer1(self):
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

        Conv = ConvLayer(kernalSize, len(inputTensor), len(kernalWeights), strideLength)
        Conv.kernalWeights = kernalWeights
        Conv.kernalBiases = kernalBias
        Conv.usePadding = usePadding
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
    
    def test_ConvLayer2(self):
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

        Conv = ConvLayer(kernalSize, len(inputTensor), len(kernalWeights), strideLength)
        Conv.kernalWeights = kernalWeights
        Conv.kernalBiases = kernalBias
        Conv.usePadding = usePadding
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

class Test_PoolingLayer:
    def test_maxPoolingLayer(self):
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

        pool = PoolingLayer(sizeOfGrid, strideLength, "max")
        output = pool.forward(inputTensor)


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

    def test_averagePoolingLayer(self):
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

        pool = PoolingLayer(sizeOfGrid, strideLength, "ave")
        output = pool.forward(inputTensor)

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

        pool = PoolingLayer(None, None, "gap")
        output = pool.forward(inputTensor)

        expected = np.array([2.5, 1, 8.5])

        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_ActivationLayer:
    def test_activationLayer(self):
        inputTensor = np.array([
            [
                [10, -10, -1, 0],
            ]
        ]) 

        output = ActivationLayer().forward(inputTensor)

        expected = np.array([
            [
                [10, -0.1, -0.01, 0],
            ]
        ])
        
        assert expected.shape == output.shape
        assert np.allclose(expected, output)

class Test_DenseLayer:
    def test_DenseLayer_forwardPropogation_noHiddenLayer(self):
        n = Network()
        n.addLayer(2)
        n.addLayer(1)

        n.weights = np.array([[0.75, 0.5]])
        inputLayer = [0.25, 0.5]
        n.biases = np.array([0.2])

        net = DenseLayer(n)
        resulting = net.forward(inputLayer)
        
        assert np.allclose(resulting, np.array([0.6375]))

    def test_DenseLayer_forwardPropogation_withHiddenLayer(self):
        n = Network()
        n.addLayer(2)
        n.addLayer(3, "relu")
        n.addLayer(2, "softmax")
        n.weights = [
            np.array([[0.75, 0.25, 0.1], [0.5, 0.75, 0.2]]),
            np.array([[0.5, 0.2], [0.4, 0.1], [0.3, 0.6]])
        ]
        n.biases = [
            np.array([0.2, 0.3, 0.1]),
            np.array([0.4, 0.5])
        ]
        inputLayer = [0.25, 0.5]

        #0.25 * 0.75 + 0.5 * 0.5 + 0.2 = 0.6375
        #0.25 * 0.25 + 0.5 * 0.75 + 0.3 = 0.7375
        #0.25 * 0.1 + 0.5 * 0.2 + 0.1 = 0.225

        hidden = np.maximum(0, np.array([0.6375, 0.7375, 0.225]))
        out = np.dot(hidden, n.weights[1]) + n.biases[1]
        output = np.exp(out) / np.sum(np.exp(out))

        net = DenseLayer(n)
        resulting = net.forward(inputLayer)

        assert np.allclose(resulting, output)

class Test_CNNModel:
    def test_CNNModel(self):
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

        # Define the dense layer
        net = Network()  
        net.addLayer(1)
        net.addLayer(1)

        net.weights = [0.5]
        net.biases = [1]

        Conv = ConvLayer(2, 4, 1, 2, padding = "no")
        Conv.kernalWeights = kernalWeights
        Conv.kernalBiases = kernalBias

        # Define the CNN model
        CNN = CNNModel(net)
        CNN.addLayer(Conv)
        CNN.addLayer(ActivationLayer())
        CNN.addLayer(PoolingLayer(2, 1, "max"))
        CNN.addLayer(DenseLayer(net))

        allTensors = CNN.forward(inputTensor)

        '''
        After Conv Layer: [
            [32, 32],
            [32, 32],
        ]
        
        After Activation Layer: [
            [32, 32],
            [32, 32],
        ]

        After Max Pooling Layer: [
            32
        ]

        Dense Layer: [
            32 * 0.5 + 1 = 17
        ]
        '''

        expectedTensor = [
            inputTensor,
            np.array([[
                [32, 32],
                [32, 32],
            ]]),
            np.array([[
                [32, 32],
                [32, 32],
            ]]),
            np.array([
                [[32]],
            ]),
            np.array([
                17,
            ])

        ]

        for i in range(len(allTensors)):
            assert np.array(allTensors[i]).shape == expectedTensor[i].shape 
            assert np.allclose(np.array(allTensors[i]), expectedTensor[i])




