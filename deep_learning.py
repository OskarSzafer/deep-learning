import numpy as np

class NN:
    # ACTIVATION FUNCTIONS
    class ReLU:
        def function(self, x):
            return np.maximum(0, x)

        def derivative(self, x):
            return np.where(x <= 0, 0, 1)

    class Sigmoid:
        def function(self, x):
            return 1 / (1 + np.exp(-x))

        def derivative(self, x):
            return x * (1 - x)

    def __init__(self, input_size, hidden_size = 10, hidden_quantity = 1, output_size = 1, activation = 'relu'):
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

        functions = {
            'relu': self.ReLU,
            'sigmoid': self.Sigmoid
        }

        self.activation = functions[activation]()
        self.activation.function = np.vectorize(self.activation.function)
        self.activation.derivative = np.vectorize(self.activation.derivative)


    def run(self, input_vector):
        self.input_layer = input_vector

        self.hidden_layer[0] = self.activation.function(np.dot(self.input_to_hidden, self.input_layer) + self.hidden_layer_bias[0])

        for i in range(1, self.hidden_quantity):
            self.hidden_layer[i] = self.activation.function(np.dot(self.hidden_to_hidden[i - 1], self.hidden_layer[i - 1]) + self.hidden_layer_bias[i])

        self.output_layer = self.activation.function(np.dot(self.hidden_to_output, self.hidden_layer[-1]) + self.output_layer_bias)

        return self.output_layer
        

    def backpropagation(self, input_vector, target_vector):
        self.run(input_vector)

        self.hidden_layer_error = np.zeros((self.hidden_quantity, self.hidden_size))
        self.output_layer_error = np.zeros(self.output_size)

        self.input_to_hidden_error = np.zeros((self.hidden_size, self.input_size))
        self.hidden_to_hidden_error = np.zeros((self.hidden_quantity - 1, self.hidden_size, self.hidden_size))
        self.hidden_to_output_error = np.zeros((self.output_size, self.hidden_size))

        # calculate neuron errors
        self.output_layer_error = (target_vector - self.output_layer)**2
        self.hidden_layer_error[-1] = np.dot(self.hidden_to_output.T, self.output_layer_error) * self.activation.derivative(self.hidden_layer[-1])

        for i in range(-2, self.hidden_quantity, -1):
            self.hidden_layer_error[i] = np.dot(self.hidden_to_output.T, self.hidden_layer_error[i+1]) * self.activation.derivative(self.hidden_layer[i])

        # calculate weight errors
        self.hidden_to_output_error = np.dot(self.hidden_layer[-1].reshape(-1, 1), self.output_layer_error.reshape(1, -1)) * self.activation.derivative(self.output_layer)
        for i in range(-2, self.hidden_quantity, -1):
            self.hidden_to_hidden_error[i] = np.dot(self.hidden_layer[i].reshape(-1, 1), self.hidden_layer_error[i+1].reshape(1, -1)) * self.activation.derivative(self.hidden_layer[i])

        self.input_to_hidden_error = np.dot(self.input_layer.reshape(-1, 1), self.hidden_layer_error[0].reshape(1, -1)) * self.activation.derivative(self.hidden_layer[0])


    def train(self, input_vectors, target_vectors, learning_rate = 0.1, epochs = 100):
        self.hidden_layer_cumulative = np.zeros((self.hidden_quantity, self.hidden_size))
        self.output_layer_cumulative = np.zeros(self.output_size)

        self.input_to_hidden_cumulative = np.zeros((self.hidden_size, self.input_size))
        self.hidden_to_hidden_cumulative = np.zeros((self.hidden_quantity - 1, self.hidden_size, self.hidden_size))
        self.hidden_to_output_cumulative = np.zeros((self.output_size, self.hidden_size))

        set_size = len(input_vectors)

        for i in range(set_size):
            self.backpropagation(input_vectors[i], target_vectors[i])

            self.hidden_layer_cumulative += self.hidden_layer_error
            self.output_layer_cumulative += self.output_layer_error

            self.input_to_hidden_cumulative += self.input_to_hidden_error.T
            self.hidden_to_hidden_cumulative += self.hidden_to_hidden_error
            self.hidden_to_output_cumulative += self.hidden_to_output_error.T

            print(f'epoch {i+1} of {set_size}')

        self.hidden_layer_bias += learning_rate * self.hidden_layer_cumulative / set_size
        self.output_layer_bias += learning_rate * self.output_layer_cumulative / set_size

        self.input_to_hidden += learning_rate * self.input_to_hidden_cumulative / set_size
        self.hidden_to_hidden += learning_rate * self.hidden_to_hidden_cumulative / set_size
        self.hidden_to_output += learning_rate * self.hidden_to_output_cumulative / set_size

        print(f'finished training {set_size} epochs')

            




nn = NN(2, 3, 2, 2)
nn.run(np.array([1, 0]))
nn.backpropagation(np.array([1, 0]), np.array([1, 0]))
nn.train([np.array([1, 0]), np.array([0, 1])], [np.array([1, 0]), np.array([0, 1])])

# a = np.array([1, 0])
# b = np.array([[1, 2, 3],[4, 5, 6]])
# print(np.dot(a, b))