import numpy as np


# activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


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

        self.z1 = self.W1 @ x + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.output = sigmoid(self.z2)

        return self.output


    # backpropagation
    def backward(self, x, target):

        prediction = self.output

        delta2 = (prediction - target) * sigmoid_derivative(prediction)

        dW2 = delta2.reshape(-1, 1) @ self.a1.reshape(1, -1)
        db2 = delta2

        delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.a1)

        dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = delta1

        # update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    # training loop
    def train(self, X, y, epochs):

        for epoch in range(epochs):

            total_loss = 0

            for i in range(len(X)):

                x = X[i]
                target = y[i]

                prediction = self.forward(x)

                loss = 0.5 * (prediction - target) ** 2
                total_loss += loss

                self.backward(x, target)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss[0]:.6f}")


    # prediction function
    def predict(self, x):
        return self.forward(x)


# -----------------------
# USE THE NETWORK
# -----------------------

# XOR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# create network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)


# train
nn.train(X, y, epochs=10000)


# test
print("\nFinal predictions:")

for x in X:
    prediction = nn.predict(x)
    print(f"{x} -> {prediction[0]:.6f}")
