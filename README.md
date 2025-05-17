# Neural Network Library

The **Neural Network Library** is a python bases library designed for building and training neural networks and convolutional networks from scratch. It offers foundational implementations of key components such as forward propagtion, backpropgation and optimisation algorithms, without relying on machine learning frameworks like TensorFlow or Pytorch

## Why this Library?

This project was developed to:

-   **Deepen understanding** of neural network by implementing them from scratch
-   **Provide a light weight alternative** to large scale frameworks for educational purposes
-   **Offer flexibility** for experimentation with custom architectures

## Feutures

**1. Custom Implementation:**
-   Fully handwritten implementations for neural network layers, activation functions and loss functions.
-   No reliance on external libraries for machine learning (except for numpy)

**2. Core neural network functionality:**
-   Support for common activation functions (eg.Leaky ReLU, Sigmoid, Softmax)
-   Multiple loss functions with derivatives (eg. MSE, MAE, Cross entropy)

**3. Training:**
-   includes backpropagation for gradient calculation and parameter updates
-   ability to experiment with different optimisation techniques.

## Roadmap
- [X] **Forward propagation**
    Implemented the feed forward pass for neural network layers
- [X] **Activation functions**
    Added support for Leaky ReLU, Sigmoid, Softmax and others
- [X] **Loss functions**
    Implemented MSE, MAE and cross entropy loss with their derivatives
- [X] **Backpropagation**
    Completed backpropagation for gradient caculation and paramter updates
- [X] **Optimisers**
    Added support for batching, stochastic gradient descent and gradient descent
- [X] **Convulational Neural Network**
    Implemented kernals, pooling and dense layers for Convulational Neural Network
- [ ] **Skin Lesion detector**    
    use the neural network library to create a model for detecting skin lesions
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

# Define a neural network achitecture
n = Network(
    lossFunc = "cross entropy",
    learningRate = 0.01
    optimisationFunc = "sgd" #schostatic gradient descent
)

n.addLayer(3, "relu") # Input layer
n.addLayer(2, "relu") # Hidden layer
n.addLayer(1, "softmax") # Output layer
n.createWeightsAndBiases()

# Train the network
accauracy, averageLoss = n.train(inputData, Labels, epochs = 10)

# Evaluate
print(f"Accauracy: {accauracy}%")
print(f"Average loss: {averageLoss}")
```