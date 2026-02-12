import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)


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

def softmax(x):
    x = np.asarray(x)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def one_hot(y, num_classes=10):
    y = np.asarray(y, dtype=int)
    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1
    return result

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
            # W = np.random.randn(output_size, input_size)
            # b = np.random.randn(output_size)
            # W = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size) # He initialization
            W = np.random.randn(output_size, input_size) * np.sqrt(2 /input_size)
            b = np.zeros(output_size)

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
                A = softmax(Z)
            else:
                A = relu(Z)

            self.activations.append(A)

        return A

    # backpropagation
    def backward(self, X, y):
        
        batch_size = X.shape[0]

        delta = self.activations[-1] - y

        for i in reversed(range(len(self.weights))):

            A_prev = self.activations[i]

            W_current = self.weights[i]

            dW = (delta.T @ A_prev) / batch_size
            db = np.mean(delta, axis=0)

            if i > 0:
                delta = (delta @ W_current) * relu_derivative(A_prev)

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    # training loop
    def train(self, X, y, epochs, batch_size=64):

        for epoch in range(epochs):
            
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for i in range(0, len(X), batch_size):

                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                prediction = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # forward pass on entire batch
            prediction = self.forward(X)

            # calculate loss for entire batch
            # loss = 0.5 * np.mean((prediction - y) ** 2)
            loss = -np.mean(np.sum(y * np.log(prediction + 1e-8), axis=1))

            if epoch % 1 == 0:
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
# X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
# y = np.sin(X)

# MNIST dataset
# convert to NumPy to avoid pandas indexing/keepdims issues
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(np.int32)

y_onehot = one_hot(y)

X_train = X[:60000]
y_train = y_onehot[:60000]

X_test = X[60000:]
y_test = y_onehot[60000:]

# create network
# nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
#nn = NeuralNetwork(input_size=1, hidden_size=100, output_size=1, learning_rate=0.01)
nn = NeuralNetwork([784, 256, 128, 10], learning_rate=0.01)

# train
nn.train(X_train, y_train, epochs=20)


# test
# print("\nFinal predictions:")

# get predictions for test data
predictions = nn.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_labels == true_labels)

print(f"Test Accuracy: {accuracy:.4f}")

# plot
# plt.figure(figsize=(10, 6))
# plt.plot(X, y, "b-", label="Actual sine wave", linewidth=2)
# plt.plot(X, predictions, "r--", label="Neural network prediction", linewidth=2)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Sine Wave: Actual vs Neural Network Prediction")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# show a grid of test images with predicted and true labels

N = 20  # total images to display
cols = 5
rows = math.ceil(N / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
axes = axes.ravel()

for i in range(N):
    img = X_test[i].reshape(28, 28)
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"P:{predicted_labels[i]} T:{true_labels[i]}")
    axes[i].axis("off")

for j in range(N, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
