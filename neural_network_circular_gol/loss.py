import numpy as np


class Loss:
    """
    Base Loss class which represent the loss calculations (includes regularization loss calculation).
    """

    def regularization_loss(self, layer):
        """
        Calculate the regularization loss of a given layer.

        :param layer: Layer to calculate regularization loss on.
        :return: Regularization loss of the layer given.
        """

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                                   np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(layer.weights *
                                          layer.weights)

        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                                   np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(layer.biases *
                                          layer.biases)

        return regularization_loss

    def calculate(self, output, classifications):
        """
        Calculates the data losses given model output and classification truth values.

        :param output: The model output.
        :param classifications: The correct classifications values to compare with the model output.
        :return: Data loss of the model output.
        """

        # Calculate sample losses
        sample_losses = self.forward(output, classifications)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class LossBinaryCrossEntropy(Loss):
    """
    Binary cross-entropy loss calculations.
    Used for binary classifications loss calculations for binary classifier model.
    """

    def forward(self, out_pred, out_true):
        """
        Calculates the sample losses between model predictions and truth (classifications) values.

        :param out_pred: Output values of the model.
        :param out_true: Truth values which should be predicted by the model.
        :return: Losses of the model.
        """

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(out_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(out_true * np.log(y_pred_clipped) +
                          (1 - out_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, dout_true):
        """
        Backward pass calculation of the gradient to apply backpropagation fixes.
        Stores the results in dinputs property.

        :param dvalues: The derived values of the samples.
        :param dout_true: The derived output truth values.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(dout_true / clipped_dvalues -
                         (1 - dout_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

