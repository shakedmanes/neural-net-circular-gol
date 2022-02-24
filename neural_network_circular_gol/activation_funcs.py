"""
This file contains activation functions for neural network.
"""
import numpy as np



class ActivationReLU:
    """
    ReLU Activation function which includes forward and backward calculation.
    """

    def forward(self, inputs):
        """
        Forward pass the input from layer calculations with the activation calculations of this function.
        Stores in the output parameter the output of the activation.

        :param inputs: The input received from a layer.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Backward pass the needed changes by the derived values (As in the backpropagation algorithm).

        :param dvalues: The derived values from the next layer (Which received from the backpropagation algorithm).
        """
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationSigmoid:
    """
    Sigmoid Activation function which includes forward and backward calculation.
    """

    # Forward pass
    def forward(self, inputs):
        """
        Forward pass the input from layer calculations with the activation calculations of this function.
        Stores in the output parameter the output of the activation.

        :param inputs: The input received from a layer.
        """
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        """
        Backward pass the needed changes by the derived values (As in the backpropagation algorithm).

        :param dvalues: The derived values from the next layer (Which received from the backpropagation algorithm).
        """
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
