import numpy as np


# Adam optimizer
class OptimizerADAM:
    """
    Adaptive Momentum Stochastic Gradient Decent Optimizer.
    """

    def __init__(
            self,
            learning_rate=0.001,
            decay=0.,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
    ):
        """
        Initialize ADAM optimizer parameters.

        :param learning_rate: The learning rate of the optimizer.
        :param decay: Decay value of the optimizer
        :param epsilon: Value for epsilon portion for the optimizer.
        :param beta_1: Beta1 value of the optimizer.
        :param beta_2: Beta2 value of the optimizer.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """
        Configure the needed configurations before updating the parameters of the layers.
        Call once before any parameter updates, to update the current learning rate based on the decay and iterations.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the parameters of a given layer.

        :param layer: Layer to update it's parameters.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = \
            self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = \
            self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = \
            layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = \
            layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = \
            self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = \
            self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = \
            layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = \
            layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += \
            -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += \
            -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """
        Post update parameters configurations.
        Call once after any parameter updates, to update the iterations number after done updating parameters.
        """
        self.iterations += 1
