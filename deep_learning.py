import numpy as np

class NN:
    def __init__(self, input_size, hidden_size = 10, hidden_quantity = 1, output_size = 1, activation = 'ReLU'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_quantity = hidden_quantity
        self.output_size = output_size

        self.input_layer = np.zeros(self.input_size)
        self.hidden_layer = np.zeros((self.hidden_quantity, self.hidden_size))
        self.output_layer = np.zeros(self.output_size)

        self.hidden_layer_bias = np.full((self.hidden_quantity, self.hidden_size), 0.1)
        self.output_layer_bias = np.full(self.output_size, 0.1)

        self.input_to_hidden = np.random.randn(self.hidden_size, self.input_size)
        self.hidden_to_hidden = np.random.randn(self.hidden_quantity - 1, self.hidden_size, self.hidden_size)
        self.hidden_to_output = np.random.randn(self.output_size, self.hidden_size)

        # print(self.input_layer)    
        # print(self.hidden_layer)
        # print(self.output_layer)

        # print(self.hidden_layer_bias)
        # print(self.output_layer_bias)

        # print(self.input_to_hidden)
        # print(self.hidden_to_hidden)
        # print(self.hidden_to_output)


nn = NN(2, 3, 2, 1)