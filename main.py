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
    
    def get_outputs(self):
        return np.array([
            neuron.output
            for neuron in self.neurons
        ])

# --------------------
# CREATE THE NETWORK
# --------------------

# create two layers
hidden_layer = Layer(num_neurons=2, num_inputs_per_neuron=2)
output_layer = Layer(num_neurons=1, num_inputs_per_neuron=2)

# training data
dataset = [
    (np.array([0,0]), 0),
    (np.array([0,1]), 1),
    (np.array([1,0]), 1),
    (np.array([1,1]), 0),
]

learning_rate = 0.5

# --------------------
# TRAINING LOOP
# --------------------

for epoch in range(10_000):
    
    total_loss = 0

    for inputs, target in dataset:
        
        # forward pass
        hidden_output = hidden_layer.forward(inputs)
        output = output_layer.forward(hidden_output)

        prediction = output[0]

        # calculate loss (mean squared error)
        loss = 0.5 * (prediction - target) ** 2
        total_loss += loss

        # backpropagation for output layer
        output_neuron = output_layer.neurons[0]

        delta_output = (
            (prediction - target)
            * sigmoid_derivative(prediction)
        )

        # store old weights before updating
        old_weights = output_neuron.weights.copy()

        # update weights and bias for output neuron
        output_neuron.weights -= (
            learning_rate
            * delta_output
            * output_neuron.inputs
        )

        output_neuron.bias -= learning_rate * delta_output

        # backpropagation for hidden layer

        for i, hidden_neuron in enumerate(hidden_layer.neurons):
            delta_hidden = (
                delta_output
                * old_weights[i]
                * sigmoid_derivative(hidden_neuron.output)
            )

            # update weights and bias for hidden neuron
            hidden_neuron.weights -= (
                learning_rate
                * delta_hidden
                * hidden_neuron.inputs
            )

            hidden_neuron.bias -= learning_rate * delta_hidden

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        # print(f"Prediction: {prediction:.6f}")
        print(f"Loss: {loss:.6f}")
        print("---------------------")

# --------------------
# FINAL LOOP
# --------------------

print('\nFinal predictions after training:')

for inputs, target in dataset:
    hidden_output = hidden_layer.forward(inputs)
    final_output = output_layer.forward(hidden_output)

    print(f'Input: {inputs}, Target: {target}, Prediction: {final_output[0]:.6f}')