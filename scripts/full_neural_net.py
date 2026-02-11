import numpy as np
import matplotlib.pyplot as plt


# activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(output):
    return (output > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(output):
    return 1 - output**2


class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate=0.1):

        self.layer_sizes = layer_sizes
        self.lr = learning_rate

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):

            # store sizes
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]

            # initialise weights and biases
            W = np.random.randn(output_size, input_size)
            b = np.random.randn(output_size)

            self.weights.append(W)
            self.biases.append(b)

    # forward propagation
    def forward(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.activations = [X]
        self.z_values = []

        A = X

        for i in range(len(self.weights)):

            W = self.weights[i]
            b = self.biases[i]

            Z = A @ W.T + b
            self.z_values.append(Z)

            # last layer = linear
            if i == len(self.weights) - 1:
                A = Z
            else:
                A = tanh(Z)

            self.activations.append(A)

        return A

    # backpropagation
    def backward(self, X, y):
        
        batch_size = X.shape[0]

        delta = self.activations[-1] - y

        for i in reversed(range(len(self.weights))):

            A_prev = self.activations[i]

            dW = (delta.T @ A_prev) / batch_size
            db = np.mean(delta, axis=0)

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

            if i > 0:
                delta = (delta @ self.weights[i]) * tanh_derivative(A_prev)

    # training loop
    def train(self, X, y, epochs):

        for epoch in range(epochs):

            # forward pass on entire batch
            prediction = self.forward(X)

            # calculate loss for entire batch
            loss = 0.5 * np.mean((prediction - y) ** 2)

            # backward pass on entire batch
            self.backward(X, y)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # prediction function
    def predict(self, x):
        output = self.forward(x)
        # return scalar for single sample, array for batch
        return output[0] if output.shape[0] == 1 else output


# -----------------------
# USE THE NETWORK
# -----------------------

# XOR dataset
# X = np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1]
# ])

# y = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])

# sin wave dataset
X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y = np.sin(X)


# create network
# nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
#nn = NeuralNetwork(input_size=1, hidden_size=100, output_size=1, learning_rate=0.01)
nn = NeuralNetwork([1, 20, 20, 20, 1])

# train
nn.train(X, y, epochs=10_000)


# test
# print("\nFinal predictions:")

# get predictions for all training data
predictions = nn.predict(X)

# plot
plt.figure(figsize=(10, 6))
plt.plot(X, y, "b-", label="Actual sine wave", linewidth=2)
plt.plot(X, predictions, "r--", label="Neural network prediction", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine Wave: Actual vs Neural Network Prediction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
