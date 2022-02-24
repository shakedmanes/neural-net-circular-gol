

class Settings:
    # Dataset settings
    DATASET_OSCILLATORS_URL = 'https://www.conwaylife.com/w/index.php?title=Category:Oscillators'
    DATASET_OSCILLATORS_CELLS_URL_FILE = 'cache/cells_urls.npy'
    DATASET_OSCILLATORS_FILE = 'cache/dataset_osc.npy'
    DATASET_NON_OSCILLATORS_FILE = 'cache/dataset_non_osc.npy'

    DATASET_BOUNDING_BOX = 60
    DATASET_BOARD_SIZE = 250
    DATASET_MAX_GENERATION_CHECK = 400

    # Neural Network settings
    INPUT_LAYER_SIZE = DATASET_BOUNDING_BOX * DATASET_BOUNDING_BOX

    # LEARNING_RATE = 0.00035
    LEARNING_RATE = 35e-5
    LEARNING_RATE_DECAY = 4e-8

    NUM_EPOCHS = 301
    NUM_BATCHES = 8

    # Neural Layers settings
    LAYER_ONE_NEURONS = 512
    LAYER_TWO_NEURONS = 2

    LAYER_ONE_WEIGHT_L1_REGULARIZATION = 5e-4
    LAYER_ONE_BIAS_L1_REGULARIZATION = 5e-4


