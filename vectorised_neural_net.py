import numpy as np

# activation functions
def sigmoid(x):
    # squashes x to between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    # derivative of sigmoid function
    return sigmoid_output * (1 - sigmoid_output)

input_size = 2
hidden_size = 2
output_size = 1

# weight matrices
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)

W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

learning_rate = 0.5

for epoch in range(10000):

    total_loss = 0

    for i in range(len(X)):

        x = X[i]
        target = y[i]

        # print(f'W1 shape: {W1.shape}')
        # print(f'W2 shape: {W2.shape}')
        # print(f'x shape: {x.shape}')

        # forward pass

        z1 = W1 @ x + b1
        a1 = sigmoid(z1)

        z2 = W2 @ a1 + b2
        prediction = sigmoid(z2)

        loss = 0.5 * (prediction - target) ** 2
        total_loss += loss

        # backpropagation

        delta2 = (prediction - target) * sigmoid_derivative(prediction)

        dW2 = delta2.reshape(-1, 1) @ a1.reshape(1, -1)
        db2 = delta2

        delta1 = (W2.T @ delta2) * sigmoid_derivative(a1)

        dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = delta1

        # update

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0]:.6f}")

# test results

print("\nFinal predictions:")

for x in X:

    z1 = W1 @ x + b1
    a1 = sigmoid(z1)

    z2 = W2 @ a1 + b2
    prediction = sigmoid(z2)

    print(f"{x} -> {prediction[0]:.6f}")