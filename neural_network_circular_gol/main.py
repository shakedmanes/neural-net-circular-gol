from warnings import filterwarnings
import numpy as np

from settings import Settings
from neural_layer import NeuralLayer
from activation_funcs import ActivationReLU, ActivationSigmoid
from loss import LossBinaryCrossEntropy
from optimizers import OptimizerADAM
from dataset_generator import DatasetGenerator


def main():
    # Create dataset and split it to train and validation batches
    dataset = DatasetGenerator()
    dataset_batches = dataset.get_batches(num_batches=Settings.NUM_BATCHES)
    train_batches, validate_batches = np.split(dataset_batches, 2)

    # Create neuron layer with 3600 input features and 512 output values
    dense1 = NeuralLayer(
        Settings.INPUT_LAYER_SIZE,
        Settings.LAYER_ONE_NEURONS,
        weight_regularizer_l2=Settings.LAYER_ONE_WEIGHT_L1_REGULARIZATION,
        bias_regularizer_l2=Settings.LAYER_ONE_BIAS_L1_REGULARIZATION
    )

    # Create ReLU activation (to be used in the first neuron layer)
    activation1 = ActivationReLU()

    # Create second neuron layer with 512 input features (as we take output
    # of previous layer here) and 2 output values (output values)
    dense2 = NeuralLayer(Settings.LAYER_ONE_NEURONS, Settings.LAYER_TWO_NEURONS)

    # Create Sigmoid activation (to be used in the second layer)
    activation2 = ActivationSigmoid()

    # Create loss function
    loss_function = LossBinaryCrossEntropy()

    # Create optimizer
    optimizer = OptimizerADAM(learning_rate=Settings.LEARNING_RATE, decay=Settings.LEARNING_RATE_DECAY)

    print('Starting training with backpropagation algorithm using ADAM optimizer:')

    # Train in loop (Backpropagation)
    for epoch in range(Settings.NUM_EPOCHS):

        train_batch, train_classifications = dataset.extract_test_and_classification(
            np.random.choice(train_batches)
        )
        train_classifications = train_classifications.reshape(-1, 1)

        # Perform a forward pass of our training data through this layer
        dense1.forward(train_batch)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)

        # Perform a forward pass through activation function
        # takes the output of second dense layer here
        activation2.forward(dense2.output)

        # Calculate the data loss
        data_loss = loss_function.calculate(activation2.output, train_classifications)
        # Calculate regularization penalty
        regularization_loss = \
            loss_function.regularization_loss(dense1) + \
            loss_function.regularization_loss(dense2)

        # Calculate overall loss
        loss = data_loss + regularization_loss

        # Calculate accuracy from output of activation2 and targets
        # Part in the brackets returns a binary mask - array consisting
        # of True/False values, multiplying it by 1 changes it into array
        # of 1s and 0s
        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions == train_classifications)

        if not epoch % 10:
            print(f'Epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' +
                  f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_function.backward(activation2.output, train_classifications)
        activation2.backward(loss_function.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    print('Performing validation batches on the model:')

    # Validate the model
    # Create test dataset
    for count, validate_batch in enumerate(validate_batches):
        validate_test, validate_classifications = dataset.extract_test_and_classification(validate_batch)
        validate_classifications = validate_classifications.reshape(-1, 1)

        # Perform a forward pass of our testing data through this layer
        dense1.forward(validate_test)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)

        # Perform a forward pass through activation function
        # takes the output of second dense layer here
        activation2.forward(dense2.output)

        # Calculate the data loss
        loss = loss_function.calculate(activation2.output, validate_classifications)

        # Calculate accuracy from output of activation2 and targets
        # Part in the brackets returns a binary mask - array consisting of
        # True/False values, multiplying it by 1 changes it into array
        # of 1s and 0s
        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions == validate_classifications)

        print(f'Validation {count}, acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == '__main__':
    # Filter annoying warnings
    filterwarnings('ignore')
    main()
