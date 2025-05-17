# Neural Network Library

The Neural Network Library is a python based library designed for creating and training neural network and convolutional neural networks. It focuses on providing a foundational implementation of key components such as forward propagation, backpropagation and optimisation algorithms, without relying on machine learning libraries like TensorFlow or PyTorch.

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
n = Network(lossFunc = "cross entropy", learningRate = 0.01)
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