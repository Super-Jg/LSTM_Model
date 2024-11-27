import numpy as np
import tensorflow as tf

class Dense:

    def __init__(self, layers):
        self.layers = layers  # layers must be an array, eg:[input layer, hidden layer, ..., hidden layer, output layer]
        self.dimension = len(layers)
        self.weights = []
        self.bias = []

    def connect(self, input, type=None):

        if input.shape[1] != self.layers[0]:
            return "Damn! The first element of dimension(array) must be the same as the column of your input."

        # initialize weights and bias
        for i in range(0, self.dimension - 1):
            self.weights.append(
                np.random.randn(self.layers[i], self.layers[i + 1])
            )
            self.bias.append(
                np.zeros([1, self.layers[i + 1]])
            )

        if type == None:
            output = input @ self.weights[0] + self.bias[0]
            for i in range(1, self.dimension - 1):
                output = output @ self.weights[i] + self.bias[i]


        elif type == "sigmoid":
            output = input @ self.weights[0] + self.bias[0]
            for i in range(1, self.dimension - 1):
                output = output @ self.weights[i] + self.bias[i]
                output = tf.nn.sigmoid(output)


        elif type == "tanh":
            output = input @ self.weights[0] + self.bias[0]
            for i in range(1, self.dimension - 1):
                output = output @ self.weights[i] + self.bias[i]
                output = tf.nn.tanh(output)


        elif type == "relu":
            output = input @ self.weights[0] + self.bias[0]
            for i in range(1, self.dimension - 1):
                output = output @ self.weights[i] + self.bias[i]
                output = tf.nn.relu(output)


        else:
            return "type not supported"

        return output


#eg:
input = np.array([[1, 2, 3],
                  [4, 5, 6]])
dense = Dense([3, 4, 5, 2])
output = dense.connect(input, type='sigmoid')
print(output)