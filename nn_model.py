import numpy as np


def mutate_weights(weights_and_biases, mutation_rate, mutation_strength):
    mutated_weights_and_biases = []
    for array in weights_and_biases:
        mutation_mask = np.random.rand(*array.shape) < mutation_rate
        mutation_values = np.random.randn(*array.shape) * mutation_strength

        mutated_array = np.where(mutation_mask, array + mutation_values, array)
        mutated_weights_and_biases.append(mutated_array)
    return mutated_weights_and_biases


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # input --> hidden
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.ones((1, hidden_size))

        # hidden --> output
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.ones((1, output_size))

        # prepare for activation, apply weights and bias
        self.hidden_layer_prepare = None
        self.output_layer_prepare = None

        # outputs
        self.hidden_layer_output = None
        self.output_layer_output = None

        self.binary_output = None

    def forward_pass(self, data):
        self.hidden_layer_prepare = np.dot(data, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_layer_output = relu(self.hidden_layer_prepare)

        self.output_layer_prepare = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        self.output_layer_output = sigmoid(self.output_layer_prepare)

        self.binary_output = (self.output_layer_output > 0.5).astype(int).flatten()

    def get_binary_output(self):
        return self.binary_output

    def get_weights(self):
        return [
            self.weights_input_hidden,
            self.weights_hidden_output,
            self.bias_input_hidden,
            self.bias_hidden_output
        ]

    def set_weights(self, weight_and_biases):
        self.weights_input_hidden = weight_and_biases[0]
        self.weights_hidden_output = weight_and_biases[1]
        self.bias_input_hidden = weight_and_biases[2]
        self.bias_hidden_output = weight_and_biases[3]

    def debug_forward_pass(self):
        print("Binary Output:", self.output_layer_output)


