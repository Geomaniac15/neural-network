# Neural Network from Scratch (NumPy)

This project implements a fully functional feedforward neural network from scratch using only NumPy. No machine learning libraries such as PyTorch, TensorFlow, or scikit-learn are used.

The goal of this project was to understand the internal mechanics of neural networks, including forward propagation, backpropagation, gradient descent, and vectorised training.

---

# Features

- Fully connected feedforward neural network
- Backpropagation implemented manually using calculus and the chain rule
- Gradient descent optimisation
- Vectorised matrix operations for efficient training
- Supports nonlinear activation functions (tanh, sigmoid, ReLU)
- Supports regression tasks (e.g. approximating sin(x))
- Modular NeuralNetwork class implementation

---

# Example: Learning the sine function

The network successfully learns to approximate the sine function:

Input:

    x ∈ [-π, π]

Target:

    y = sin(x)

Result:

The network produces a smooth approximation of the sine wave after
training.

This demonstrates that the network can learn nonlinear continuous
functions.

---

# Architecture

The network uses a standard feedforward architecture:

    Input layer      size = input_size
    Hidden layer     size = hidden_size, activation = tanh
    Output layer     size = output_size, activation = linear

Mathematically:

    Z1 = W1 · X + b1
    A1 = tanh(Z1)

    Z2 = W2 · A1 + b2
    Output = Z2

---

# Training method

The network is trained using gradient descent.

Loss function:

    Mean Squared Error (MSE)

Backpropagation computes gradients using the chain rule:

    ∂Loss/∂W
    ∂Loss/∂b

Weights are updated using:

    W = W − learning_rate × gradient

---

# Why this project exists

Modern ML frameworks hide the underlying mechanics of neural networks.
This project rebuilds those mechanics from first principles to develop a
deeper understanding of:

-   how neural networks actually compute outputs
-   how backpropagation works
-   how gradient descent updates weights
-   how activation functions affect learning
-   how matrix operations power neural networks

---

# Example usage

``` python
nn = NeuralNetwork(input_size=1, hidden_size=100, output_size=1, learning_rate=0.01)

nn.train(X, y, epochs=10000)

prediction = nn.predict(x)
```

---

# Requirements

    numpy
    matplotlib

Install with:

    pip install numpy matplotlib

---

# Future improvements

-   Fully vectorised batch training
-   Multiple hidden layers (deep networks)
-   MNIST digit classification
-   Saving and loading trained models
-   Additional activation functions
