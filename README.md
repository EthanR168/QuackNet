# QuackNet

The **QuackNet** is a python based library designed for building and training neural networks and convolutional networks from scratch. It offers foundational implementations of key components such as forward propagation, backpropagation and optimisation algorithms, without relying on machine learning frameworks like TensorFlow or Pytorch

## Why this Library?

This project was developed to:

-   **Deepen understanding** of neural network by implementing them from scratch
-   **Provide a lightweight alternative** to large scale frameworks for educational purposes
-   **Offer flexibility** for experimentation with custom architectures

## Features

**1. Custom Implementation:**
-   Fully handwritten implementations for neural network layers, activation functions and loss functions.
-   No reliance on external libraries for machine learning (except for numpy)

**2. Core neural network functionality:**
-   Support for common activation functions (eg.Leaky ReLU, Sigmoid, Softmax)
-   Multiple loss functions with derivatives (eg. MSE, MAE, Cross entropy)

**3. Training:**
-   includes backpropagation for gradient calculation and parameter updates
-   ability to experiment with different optimization techniques.

## Roadmap
- [X] **Forward propagation**
    Implemented the feed forward pass for neural network layers
- [X] **Activation functions**
    Added support for Leaky ReLU, Sigmoid, Softmax and others
- [X] **Loss functions**
    Implemented MSE, MAE and cross entropy loss with their derivatives
- [X] **Backpropagation**
    Completed backpropagation for gradient calculation and parameter updates
- [X] **Optimisers**
    Added support for batching, stochastic gradient descent and gradient descent
- [X] **Convolutional Neural Network**
    Implemented kernels, pooling and dense layers for Convolutional Neural Network
- [ ] **Benchmark against PyTorch/TensorFlow**
    Benchmark against popular machine learning framworks on MNIST library
- [ ] **Skin Lesion detector**    
    use the neural network library to create a model for detecting skin lesions using HAM10000 for skin lesion images
- [ ] **Add Adams optimiser**  
    Implement the Adam optimiser to improve training performance and convergence
- [ ] **Additional activation functions**  
    implement advanced activation functions (eg. GELU and Swish)
- [ ] **Visualisation tools**  
    add support for visualising training, such as loss and accuracy graphs

## Usage
Here is an example of how to create and train a simple neural network using the library:
```python
from neuralLibrary.main import Network

# Define a neural network architecture
n = Network(
    lossFunc = "cross entropy",
    learningRate = 0.01
    optimisationFunc = "sgd" #stochastic gradient descent
)
n.addLayer(3, "relu") # Input layer
n.addLayer(2, "relu") # Hidden layer
n.addLayer(1, "softmax") # Output layer
n.createWeightsAndBiases()

# Example data
inputData = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
labels = [[1], [0]]

# Train the network
accuracy, averageLoss = n.train(inputData, Labels, epochs = 10)

# Evaluate
print(f"Accuracy: {accuracy}%")
print(f"Average loss: {averageLoss}")
```

## Examples

-   [Simple Neural Network Example](/ExampleCode/NNExample.py): A basic neural network implementation demonstrating forward and backpropagation
-   [Convolutional Neural Network Example](/ExampleCode/CNNExample.py): Shows how to use the convolutional layers in the library
-   [MNIST Neural Network Example](/ExampleCode/MNISTExample/mnistExample.py): Shows how to use neural network to train on MNIST

## Code structure

## Neural Network Class
-   **Purpose** Handles fully connected layers for standard neural network
-   **Key Componets:**
    -   Layers: Dense Layer
    -   Functions: Forward propagation, backpropgation
    -   Optimisers: SGD, GD, GD using batching

## Convolutional Neural Network Class
-   **Purpose** Specialised for image data processing using covolutional layers
-   **Key Components:**
    -   Layers: Convolutional, pooling, dense and activation layers
    -   Functions: Forward propagation, backpropgation
    -   Optimsers: SGD, GD, GD using batching