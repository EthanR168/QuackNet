from .CNN import CNNModel
from .CNN import Conv1DLayer, Conv2DLayer, ActivationLayer, DenseLayer, PoolingLayer, GlobalAveragePooling

from .Transformer import EmbeddingLayer, FeedForwardNetwork, MultiAttentionHeadLayer, NormLayer, PositionalEncoding, ResidualConnection
from .Transformer import TransformerBlock, Transformer

from .RNN import StackedRNN, SingularRNN

from .core.losses.lossFunctions import *
from .core.losses.lossDerivativeFunctions import *

from .core.activations.activationFunctions import *
from .core.activations.activationDerivativeFunctions import *

from .core.utilities.dataAugmentation import *
from .core.utilities.drawGraphs import *

from .core.optimisers.adam import Adam
from .core.optimisers.adamW import AdamW
from .core.optimisers.stochasticGD import SGD
from .core.optimisers.gradientDescent import GD
from .core.optimisers.rmsProp import RMSProp
from .core.optimisers.lion import Lion

from .NN import Network

from .universalLayers import Dropout

"""
# QuackNet

**QuackNet** is a from scratch deep learning library built entirely in NumPy. 
It supports Neural Networks, CNNs, RNNs, and Transformers, providing full access to forward 
and backward propagation, manual gradient computation, and weight updates. All without relying 
on frameworks like TensorFlow or PyTorch.

QuackNet is designed for both educational purposes and small-scale research, letting you 
experiment with deep learning architectures while understanding the underlying math.

## Key Features

**1. Core Functionality:**
-   Forward and backward propagation for NN, CNN, RNN, and Transformer models
-   Fully customisable architectures
-   Manual gradient computation for each layer
-   Modular API design

**2. Supported Layers:**
-   Dense (fully connected) layers
-   Convolutional layers (1D and 2D)
-   Pooling (Max and Average, Global Average)
-   Multi head self attention, positional encoding, residiual connections

**3. Activation and Loss Functions:**
-   Activation functions: ReLU, Leaky ReLU, Sigmoid, TanH, Linear, Softmax with derivatives
-   Loss functions: MSE, MAE, Cross entropy, Normalised Cross Entropy with derivatives

**4. Optimisers:**
-   Gradient Descent (GD)
-   Stochastic Gradiend Descent (SGD)
-   Adam, AdamW, RMSProp, Lion
-   Supports batching

**5. Utilities and Extras:**
-   Save/Load weights and biases
-   Training visualisation (accuracy/loss graphs)
-   Data augmentation (flipping, normalisation, one hot labels)
-   Evaluation metrics (accuracy, loss)
-   Demo projects (MNIST, HAM10000 skin lesion detection)
-   150+ unit tests with 91% coverage
-   Benchmarking against PyTorch and TensorFlow

## Installation

QuackNet is simple to install via PyPI.

**Install via PyPI**

```
pip install QuackNet
```

## Usage Example

```Python
from quacknet.main import Network

# Define a neural network architecture
n = Network(
    lossFunc = "cross entropy",
    learningRate = 0.01,
    optimisationFunc = "sgd", #stochastic gradient descent
)
n.addLayer(3, "relu") # Input layer
n.addLayer(2, "relu") # Hidden layer
n.addLayer(1, "softmax") # Output layer
n.createWeightsAndBiases()

# Example data
inputData = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
labels = [[1], [0]]

# Train the network
accuracy, averageLoss = n.train(inputData, labels, epochs = 10)

# Evaluate
print(f"Accuracy: {accuracy}%")
print(f"Average loss: {averageLoss}")
```

## Examples

-   [Simple Neural Network Example](/ExampleCode/NNExample.py): A basic neural network implementation demonstrating forward and backpropagation
-   [Convolutional Neural Network Example](/ExampleCode/CNNExample.py): Shows how to use the convolutional layers in the library
-   [MNIST Neural Network Example](/ExampleCode/MNISTExample/mnistExample.py): Shows how to use neural network to train on MNIST

## Highlights

-   Fully from scratch implementation for educational insight
-   Full suport for modern deep learning architecures
-   Benchmarked against PyTorch and TensorFlow on MNIST and had better performance
-   Easy to use API and modular design for experimenting

## Code structure

-   **Neural Network Class:**
    -   Dense Layers, forward/backprop
-   **Convolutional Neural Network Class:**
    -   Conv1D/2D, pooling, global average pooling, activation layers
-   **Recurrent Neural Network Class:**
    -   Singular and Stacked RNN, with BPTT
-   **Transformer Class:**
    -   Multi head attention with casual padding, position wise FFN, residual connections, input embedding layer, positional encoding
    
## Related Projects

### Skin Lesion Detector

A convolutional neural network (CNN) skin lesion classification model built with QuackNet, trained using the HAM10000 dataset. This model achieved 72% accuracy on a balanced test set.

You can explore the full project here:
[Skin Lesion Detector Repository](https://github.com/SirQuackPng/skinLesionDetector)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""