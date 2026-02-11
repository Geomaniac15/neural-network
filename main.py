import numpy as np

def sigmoid(x):
    # squashes x to between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    # derivative of sigmoid function
    return sigmoid_output * (1 - sigmoid_output)

class Neuron:

    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()

        # caches for backpropagation
        self.inputs = None
        self.z = None
        self.output = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = sigmoid(self.z)
        return self.output

class Layer:

    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = []
    
        for _ in range(num_neurons):
            neuron = Neuron(num_inputs_per_neuron)
            self.neurons.append(neuron)
    
    def forward(self, inputs):
        outputs = []

        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        
        return np.array(outputs)

# --------------------
# CREATE THE NETWORK
# --------------------

# create two layers
hidden_layer = Layer(num_neurons=3, num_inputs_per_neuron=5)
output_layer = Layer(num_neurons=1, num_inputs_per_neuron=3)

# training data
inputs = np.array([0, 1, 1, 1, 0])
target = np.array([1])  # we want the output to be 1 for this input

learning_rate = 0.1

# --------------------
# TRAINING LOOP
# --------------------

for epoch in range(1000):

    # forward pass
    hidden_output = hidden_layer.forward(inputs)
    output = output_layer.forward(hidden_output)

    prediction = output[0]

    # calculate loss (mean squared error)
    loss = 0.5 * (prediction - target[0]) ** 2

    # get output neuron
    output_neuron = output_layer.neurons[0]

    # backpropagation for output layer
    delta_output = (prediction - target[0]) * sigmoid_derivative(prediction)

    # gradients
    dW = delta_output * output_neuron.inputs
    dB = delta_output

    # update weights and bias for output neuron
    output_neuron.weights -= learning_rate * dW
    output_neuron.bias -= learning_rate * dB

    if epoch % 100 == 0:
        print(f"Epoch {epoch}")
        print(f"Prediction: {prediction:.6f}")
        print(f"Loss: {loss:.6f}")
        print("---------------------")

# --------------------
# FINAL LOOP
# --------------------

hidden_output = hidden_layer.forward(inputs)
final_output = output_layer.forward(hidden_output)

print(f'Final prediction after training: {final_output[0]:.6f}')