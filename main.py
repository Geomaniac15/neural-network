import numpy as np

def sigmoid(x):
    # squashes x to between 0 and 1
    return 1 / (1 + np.exp(-x))

class Neuron:

    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return sigmoid(weighted_sum)

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

# create a layer with 3 neurons, each with 5 inputs
layer = Layer(num_neurons=3, num_inputs_per_neuron=5)

# test input
inputs = np.array([0, 1, 1, 1, 0])

output = layer.forward(inputs)

print(f'Layer Output: {output}')