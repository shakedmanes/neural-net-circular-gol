from warnings import filterwarnings
from urllib.parse import urlparse
from random import randint
from re import compile as re_compile

from bs4 import BeautifulSoup
from requests import get as get_request
from seagull import Board, Simulator
from seagull.rules import conway_classic
from seagull.lifeforms import Custom
from seagull.lifeforms.wiki import parse_cells
from loguru import logger

import numpy as np

from settings import Settings


class DatasetGenerator:
    """
    Containing the management unit for dataset generation for oscillators and non oscillators.
    Pulls oscillators from the `Game of Life Official Wiki <https://www.conwaylife.com/w/index.php?title=Category:Oscillators>`_
    and generates non oscillators on the fly using randomization of configuration for the GOL and
    testing them in real game simulation.
    """

    __oscillators_base_page = Settings.DATASET_OSCILLATORS_URL
    __oscillators_base_url = f'{urlparse(__oscillators_base_page).scheme}://{urlparse(__oscillators_base_page).netloc}'
    __oscillators_bounding_box_size = Settings.DATASET_BOUNDING_BOX
    __oscillators_cells_url_file = Settings.DATASET_OSCILLATORS_CELLS_URL_FILE
    __oscillators_dataset_file = Settings.DATASET_OSCILLATORS_FILE
    __non_oscillators_dataset_file = Settings.DATASET_NON_OSCILLATORS_FILE

    __simulation_board_size = Settings.DATASET_BOARD_SIZE
    __simulation_max_generation = Settings.DATASET_MAX_GENERATION_CHECK

    def __init__(self, load_configurations=True, load_cache_urls=True):
        """
        Initializes the dataset for oscillators and non oscillators configurations.

        :param load_configurations: (Default True) Indicates whether to load the configurations from
        local store or generate them on the fly.
        :param load_cache_urls: (Default True) Indicates whether to load the oscillators cache urls or to
        scrap them from the official wiki on the fly.
        """
        if not load_configurations:

            if not load_cache_urls:
                print('Loading list of oscillators pages')
                self.__oscillators_pages_urls = self.__load_list_of_oscillators()
                print('Extracting the cells files from the oscillators pages')
                self.__oscillators_cells_urls = self.__extract_oscillators_cells_files()
                self.__save_configurations_to_file(
                    DatasetGenerator.__oscillators_cells_url_file,
                    self.__oscillators_cells_urls
                )
            else:
                print('Parsing the oscillators configurations from the cells files')
                self.__oscillators_cells_urls = \
                    self.__load_configurations_from_file(DatasetGenerator.__oscillators_cells_url_file)

            self.__oscillators_configurations = self.__convert_cells_files_to_configurations()
            print(f'Done parsing {len(self.__oscillators_configurations)} oscillators, saving them into dataset file')
            self.__save_configurations_to_file(
                DatasetGenerator.__oscillators_dataset_file,
                self.__oscillators_configurations
            )
            print('Done saving oscillators configurations dataset.')

            print(f'Generating {len(self.__oscillators_configurations)} non oscillators configurations dataset.')
            self.__non_oscillators_configurations = \
                self.__generate_non_oscillators_configurations(len(self.__oscillators_configurations))
            print(f'Saving non oscillators configurations dataset')
            self.__save_configurations_to_file(
                DatasetGenerator.__non_oscillators_dataset_file,
                self.__non_oscillators_configurations
            )
            print('Done saving non oscillators configuration dataset')

        else:
            print('Loading oscillators from existing oscillators dataset file')
            self.__oscillators_configurations = \
                self.__load_configurations_from_file(DatasetGenerator.__oscillators_dataset_file)
            print(f'Done loading {len(self.__oscillators_configurations)} oscillators configurations dataset.')

            print('Loading non oscillators from existing non oscillators dataset file')
            self.__non_oscillators_configurations = \
                self.__load_configurations_from_file(DatasetGenerator.__non_oscillators_dataset_file)
            print(f'Done loading {len(self.__non_oscillators_configurations)} non oscillators configurations dataset.')

    def get_batches(self, num_batches=5):
        """
        Returns shuffled batches including classifications for each configuration.

        :param num_batches: Number of batches to separate into.
        :return: Numpy array containing dataset batches.
        """
        # Splitting and classifying the batches
        osc_classifications, non_osc_classifications = self.__attach_classifications_to_datasets()
        split_osc_class = np.array_split(osc_classifications, num_batches)
        split_non_osc_class = np.array_split(non_osc_classifications, num_batches)
        shuffled_batches = []

        # Concatenating and shuffling the batches to create randomized batches
        for index in range(num_batches):
            created_batch = np.concatenate([split_osc_class[index], split_non_osc_class[index]])
            np.random.shuffle(created_batch)
            shuffled_batches.append(created_batch)

        return np.array(shuffled_batches)

    def __extract_oscillators_cells_files(self):
        """
        Extracts oscillators cells files urls from the oscillators wiki pages urls.

        :return: List of oscillators cells files urls.
        """
        osc_cells_file_urls = []
        index = 1

        for osc_page in self.__oscillators_pages_urls:
            print(f'Parsing oscillator {index}')
            index += 1

            cells_file_url = DatasetGenerator.__extract_cell_file_from_page(osc_page)

            if cells_file_url:
                osc_cells_file_urls.append(cells_file_url)

        return osc_cells_file_urls

    def __convert_cells_files_to_configurations(self):
        """
        Convert oscillators cells files urls to actual GOL fixed configurations.

        :return: List of oscillators configurations in fixed bounding box.
        """
        osc_shapes = []
        index = 1

        for osc_cell_file_url in self.__oscillators_cells_urls:
            print(f'Creating configuration for oscillator {index}')
            index += 1
            configuration = DatasetGenerator.__generate_configuration_from_cell_file_url(osc_cell_file_url)

            # Checking if the configuration is found and is in the right size for the bounding box
            # (Or even to be fixed into the bounding box).
            if configuration is not None and \
               configuration.shape[0] <= DatasetGenerator.__oscillators_bounding_box_size and \
               configuration.shape[1] <= DatasetGenerator.__oscillators_bounding_box_size:
                fixed_configuration = DatasetGenerator.__fix_configuration_layout(configuration)
                osc_shapes.append(fixed_configuration.reshape(DatasetGenerator.__oscillators_bounding_box_size ** 2))
            else:
                print('Passing configuration - not found or out of accepted bounding box')

        return osc_shapes

    def __attach_classifications_to_datasets(self):
        """
        Attaches classifications to oscillators and non oscillators configurations.

        :return: List containing 2 elements - oscillators and non oscillators labeled configurations.
        """
        return [
            np.array([[conf, 1] for conf in self.__oscillators_configurations]),
            np.array([[conf, 0] for conf in self.__non_oscillators_configurations])
        ]

    @classmethod
    def __generate_non_oscillators_configurations(cls, size=425):
        """
        Generates non oscillators configurations by given size.

        :param size: Number of non oscillators configuration to generate.
        :return: List of non oscillators configurations.
        """
        generated_configurations = []

        while len(generated_configurations) != size:
            print('Generating configuration')
            gen_conf = cls.__generate_configuration()
            print('Running simulation for configuration')
            simulation_history = cls.__run_simulation(gen_conf)
            print('Validating non circularly for configuration')
            if not cls.__is_circular_configuration(simulation_history):
                generated_configurations.append(gen_conf)
                print(f'Generated configuration - {len(generated_configurations)}!')

        return generated_configurations

    @classmethod
    def __load_list_of_oscillators(cls):
        """
        Loads oscillators pages from the oscillators category at the html page in the GOL wiki.

        :return: List of oscillators pages urls.
        """
        current_html_page = cls.__extract_oscillators_html_page(cls.__oscillators_base_page)
        current_next_page = cls.__extract_next_oscillators_page(current_html_page)
        oscillators_pages = []

        # Extracting the categories of oscillators from the html page
        osc_pages = current_html_page.find('div', id='mw-pages')
        osc_categories = osc_pages.find('div', attrs={'class': 'mw-category'})
        osc_categories_groups = osc_categories.find_all('h3', string=re_compile(r'[^\s]'))

        # For each category, extract the oscillator page url
        for osc_category in osc_categories_groups:
            for link in osc_category.find_next('ul').find_all(href=True):
                oscillators_pages.append(cls.__oscillators_base_url + link['href'])

        # Continue extracting the whole oscillators page urls from all categories
        while current_next_page is not None:
            current_html_page = cls.__extract_oscillators_html_page(current_next_page)
            current_next_page = cls.__extract_next_oscillators_page(current_html_page)

            osc_pages = current_html_page.find('div', id='mw-pages')
            osc_categories = osc_pages.find('div', attrs={'class': 'mw-category'})
            osc_categories_groups = osc_categories.find_all('h3', string=re_compile(r'[^\s]'))

            for osc_category in osc_categories_groups:
                for link in osc_category.find_next('ul').find_all(href=True):
                    oscillators_pages.append(cls.__oscillators_base_url + link['href'])

        return oscillators_pages

    @classmethod
    def __generate_configuration(cls):
        """
        Generate random configuration in the GOL.

        :return: Numpy array of fixed configuration in the GOL.
        """
        # Choose alive (ones) and dead (zeros) cells
        ones = randint(0, cls.__oscillators_bounding_box_size ** 2)
        zeros = (cls.__oscillators_bounding_box_size ** 2) - ones

        # Randomize and rearrange the living and dead cells
        configuration = np.array([0] * zeros + [1] * ones)
        np.random.shuffle(configuration)

        return configuration

    @classmethod
    def __run_simulation(cls, configuration):
        """
        Runs simulation of the GOL with given configuration for predefined number of generations.

        :param configuration: Numpy array of configuration to the GOL.
        :return: Simulation history of the given configuration.
        """
        # Creating a GOL board which includes the configuration in the center of the board, as matrix.
        board = Board(size=(cls.__simulation_board_size, cls.__simulation_board_size))
        board.add(
            Custom(configuration.reshape(cls.__oscillators_bounding_box_size, cls.__oscillators_bounding_box_size)),
            loc=(cls.__simulation_board_size // 2, cls.__simulation_board_size // 2)
        )

        # Simulate the configuration steps in the GOL board.
        simulator = Simulator(board)
        simulator_stats = simulator.run(
            conway_classic,
            iters=cls.__simulation_max_generation
        )

        return simulator.get_history()

    @classmethod
    def __is_circular_configuration(cls, simulation_history):
        """
        Checks if a given configuration is circular by its simulation history.

        :param simulation_history: The simulation history of some configuration.
        :return: True if the configuration is circular, Otherwise false.
        """
        _, inverse_indices = np.unique(simulation_history, axis=0, return_inverse=True)

        # Creating unique map to save already seen unique states
        unique_map = dict()
        configuration_history_length = len(inverse_indices)

        for index in range(configuration_history_length):
            unique_element_index = unique_map.get(inverse_indices[index], None)

            # Found unique element again, means there's repetition in the configuration history
            if unique_element_index is not None:
                if unique_element_index == 0:
                    return True
                return False

            # New unique value found, add it to the unique map
            else:
                unique_map[inverse_indices[index]] = index

        # No repetition found at all, whole history is unique
        return False

    @classmethod
    def __fix_configuration_layout(cls, configuration):
        """
        Fix configuration layout to fixed configuration bounding box.

        :param configuration: Configuration to fix.
        :return: Padded configuration by fixed bounding box.
        """
        # Calculating the size of padding to make the configuration be fixed into the bounding box.
        row_pad_before = (cls.__oscillators_bounding_box_size - configuration.shape[0]) // 2
        row_pad_after = cls.__oscillators_bounding_box_size - row_pad_before - configuration.shape[0]
        col_pad_before = (cls.__oscillators_bounding_box_size - configuration.shape[1]) // 2
        col_pad_after = cls.__oscillators_bounding_box_size - col_pad_before - configuration.shape[1]

        return np.pad(
            configuration,
            ((row_pad_before, row_pad_after), (col_pad_before, col_pad_after)),
            mode='constant',
            constant_values=0
        )

    @staticmethod
    def __save_configurations_to_file(file_name, configurations):
        """
        Saves configurations to file.

        :param file_name: File name to save the configurations to.
        :param configurations: Configurations to save into a file.
        """
        np.save(file_name, configurations, allow_pickle=True)

    @staticmethod
    def __load_configurations_from_file(file_name):
        """
        Loads saved configurations from file.

        :param file_name: File name to load configurations from.
        :return: The configurations which the given file holds.
        """
        return np.load(file_name, allow_pickle=True)

    @staticmethod
    def extract_test_and_classification(batch):
        """
        Extracts the tests and classifications from a given batch.

        :param batch: Batch to extract tests and classifications from.
        :return: List containing the tests in the first element and the classifications as the second element.
        """
        # Split the tests and classifications values
        tests, classifications = np.split(batch, indices_or_sections=[1], axis=1)

        # Extract the tests and classifications arrays (Because it 3 dimensional array)
        squeezed_tests = np.array([test for test in np.squeeze(tests)])
        squeezed_classifications = np.array([classification for classification in np.squeeze(classifications)])

        return [squeezed_tests, squeezed_classifications]

    @staticmethod
    def __extract_oscillators_html_page(oscillators_url):
        """
        Extracts oscillators html page from a given url.

        :param oscillators_url: Url of oscillators.
        :return: Html page by the oscillators url given.
        """
        html_text = get_request(oscillators_url).text
        return BeautifulSoup(html_text, 'html.parser')

    @staticmethod
    def __extract_next_oscillators_page(parsed_html):
        """
        Extracts next oscillators page url from a given parsed html file of oscillators list.

        :param parsed_html: Parsed html page of oscillators.
        :return: The next oscillators page url if found, Otherwise None.
        """
        urls_found = parsed_html.find_all('a', href=True, string=re_compile(r'next page'))

        if len(urls_found) > 0:
            return DatasetGenerator.__oscillators_base_url + urls_found[0]['href']

        return None

    @staticmethod
    def __extract_cell_file_from_page(url):
        """
        Extracts the cells file url from a given url.

        :param url: Url to extract cells file from.
        :return: The extracted url of the cells file if found, Otherwise None.
        """
        osc_html_page = DatasetGenerator.__extract_oscillators_html_page(url)
        cells_link = osc_html_page.find('a', string=re_compile(r'.+\.cells'))

        if cells_link is not None:
            return cells_link['href']
        return None

    @staticmethod
    def __generate_configuration_from_cell_file_url(cell_url):
        """
        Generate configuration from cell file url.

        :param cell_url: Cell url of a configuration.
        :return: Numpy array of the given configuration in the cell url. If an error occurred, return None.
        """
        try:
            configuration = parse_cells(cell_url)
            return configuration.layout
        except Exception as exc:
            print(f'Failed loading configuration due error - {exc}, configuration passed')
            return None


# Just for testing the functionality of the dataset generator
if __name__ == '__main__':
    # Disable annoying logs from the seagull library
    logger.disable('seagull')
    filterwarnings('ignore')
    dataset_generator = DatasetGenerator(load_configurations=True)
