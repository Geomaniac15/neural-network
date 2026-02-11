import numpy as np


# activation function
def sigmoid(x):
    # squashes x to between 0 and 1
    return 1 / (1 + np.exp(-x))


# inputs to the neuron
inputs = np.array([0, 1])

# weights (random initially)
weights = np.array([0.5, -0.5])

# bias
bias = 0.0

# weighted sum
weighted_sum = np.dot(weights, inputs) + bias

# output after activation
output = sigmoid(weighted_sum)

print(f"Weighted sum: {weighted_sum}")
print(f"Output: {output}")
