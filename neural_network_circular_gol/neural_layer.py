import numpy as np


class NeuralLayer:
    """
    Represent KNN Neural layer with the ability to forward and backward neurons calculations.
    """

    def __init__(
            self,
            n_inputs,
            n_neurons,
            weight_regularizer_l1=0,
            weight_regularizer_l2=0,
            bias_regularizer_l1=0,
            bias_regularizer_l2=0
    ):
        """
        Layer initialization by a given parameters.

        :param n_inputs: Number of input the layer received.
        :param n_neurons: Number of neurons the layer should have.
        :param weight_regularizer_l1: Weight for L1 regulation.
        :param weight_regularizer_l2: Weight for L2 regulation.
        :param bias_regularizer_l1: Bias for L1 regulation.
        :param bias_regularizer_l2: Bias for L2 regulation.
        """
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        """
        Forward pass the input from other layer into this layer for calculation of output.
        Stores in the output parameter the output of the layer.

        :param inputs: The input received from other layer.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        """
        Backward pass the needed changes by the derived values (As in the backpropagation algorithm).

        :param dvalues: The derived values from the next layer (Which received from the backpropagation algorithm).
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
