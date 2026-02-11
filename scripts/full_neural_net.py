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
    return 1 - output ** 2


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):

        # store sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # learning rate
        self.lr = learning_rate

        # initialise weights and biases
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size)

        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.random.randn(output_size)


    # forward propagation
    def forward(self, x):
        # x shape: (batch_size, input_size) or (input_size,) for single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.z1 = x @ self.W1.T + self.b1
        self.a1 = tanh(self.z1)

        self.z2 = self.a1 @ self.W2.T + self.b2
        self.output = self.z2

        return self.output


    # backpropagation
    def backward(self, x, target):
        # x shape: (batch_size, input_size), target shape: (batch_size, output_size)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        batch_size = x.shape[0]
        prediction = self.output

        delta2 = (prediction - target)

        dW2 = delta2.T @ self.a1 / batch_size
        db2 = np.mean(delta2, axis=0)

        delta1 = delta2 @ self.W2 * tanh_derivative(self.a1)

        dW1 = delta1.T @ x / batch_size
        db1 = np.mean(delta1, axis=0)

        # update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


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
nn = NeuralNetwork(input_size=1, hidden_size=100, output_size=1, learning_rate=0.01)

# train
nn.train(X, y, epochs=10_000)


# test
# print("\nFinal predictions:")

# get predictions for all training data
predictions = nn.predict(X)

# plot
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'b-', label='Actual sine wave', linewidth=2)
plt.plot(X, predictions, 'r--', label='Neural network prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave: Actual vs Neural Network Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
