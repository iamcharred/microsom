import unittest
import traceback
import sys

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy import ones

import microsom as msom


def green(s):
    """
    Returns the input string with green color formatting.

    Parameters:
    s (str): The input string to be formatted.

    Returns:
    str: The input string with green color formatting.
    """
    return f'\033[1;32m{s}\033[m'

def yellow(s):
    """
    Returns the input string formatted as yellow text.

    Parameters:
    s (str): The input string to be formatted.

    Returns:
    str: The input string formatted as yellow text.
    """
    return f'\033[1;33m{s}\033[m'

def red(s):
    """
    Returns the input string formatted with red color.

    Parameters:
    s (str): The input string to be formatted.

    Returns:
    str: The formatted string with red color.
    """
    return f'\033[1;31m{s}\033[m'

def log(*m):
    """
    Logs the given messages to the console.

    Parameters:
    *m: Variable number of arguments representing the messages to be logged.

    Returns:
    None
    """
    print(" ".join(map(str, m)))

def log_exit(*m):
    """
    Logs an error message and exits the program.

    Parameters:
    *m: Variable number of arguments representing the error message.

    Returns:
    None
    """
    log(red("ERROR:"), *m)
    sys.exit(1)

class TestSOM(unittest.TestCase):
    """
    A class that contains unit tests for the SOM algorithm.

    This class inherits from the `unittest.TestCase` class and defines various test methods to evaluate the SOM implementation.

    The tests include:
    - Testing the Euclidean distance calculation
    - Testing the Gaussian function
    - Testing the prediction method
    - Testing the quantization error calculation
    - Testing the decay function
    - Training the SOM

    Each test method is responsible for asserting the expected behavior of a specific aspect of the SOM algorithm.

    """

    def setUp(self):
        """
        Set up the test environment.

        This method is called before each test method is executed.
        It initializes the SOM object with fake weights and sets up the input data.

        Returns:
        None
        """
        test_name = "setUp"
        self.som = msom.SOM(4, 4, 3, sigma=0.5, learning_rate=0.5, random_seed=42)
        self.som._weights = ones((4, 4, 3))  # fake weights
        self.som._weights[1, 0] = ([0,0,0])
        self._input_data = np.array([[1,1,1],[2,2,2],[3,3,3]])
        log(green("PASS"), test_name, "")

    def test_euclidean_distance(self):
        """
        Test the Euclidean distance calculation.

        This method tests the `_compute_distance_map` method of the SOM class.
        It computes the distance map for the first input data and asserts the expected result.

        Returns:
        None
        """
        test_name = "test_euclidean_distance"
        self.som._compute_distance_map(self._input_data[0])
        d = self.som.distance_map
        assert_almost_equal(d[1,0], 1.73205080)
        log(green("PASS"), test_name, "")

    def test_gaussian(self):
        """
        Test the Gaussian function.

        This method tests the `_gaussian` method of the SOM class.
        It computes the Gaussian function for a specific position and asserts the expected result.

        Returns:
        None
        """
        test_name = "test_gaussian"
        bell = self.som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)
        log(green("PASS"), test_name, "")

    def test_predict(self):
        """
        Test the predict method.

        This method tests the `predict` method of the SOM class.
        It predicts the winning neuron for the second input data and asserts the expected result.

        Returns:
        None
        """
        test_name = "test_predict"
        assert self.som.predict(self._input_data[1]) == self.som.winner(self._input_data[1])
        log(green("PASS"), test_name, "")

    def test_quantization_error(self):
        """
        Test the quantization error calculation.

        This method tests the `quantization_error` method of the SOM class.
        It computes the quantization error for the input data and asserts the expected result.

        Returns:
        None
        """
        assert_almost_equal(self.som.quantization_error(self._input_data), 1.73205080)

    def test_decay_function(self):
        """
        Test the decay function.

        This method tests the `_asymptotic_decay` method of the SOM class.
        It computes the decay value for specific parameters and asserts the expected result.

        Returns:
        None
        """
        test_name = "test_decay_function"
        self.som.train(self._input_data, 100)
        assert self.som._asymptotic_decay(1,1) == 1 * np.exp(-1 / (100 / np.log(.5)))
        assert self.som._asymptotic_decay(1,2) == 1 * np.exp(-2 / (100 / np.log(.5)))
        log(green("PASS"), test_name, "")

    def test_train(self):
        """
        Test the SOM training.

        This method tests the `train` method of the SOM class.
        It trains the SOM with random input data and asserts the expected weights.

        Returns:
        None
        """
        test_name = "test_train"
        random_seed = 40
        np.random.seed(random_seed)
        input_data = np.random.random((20,3))
        n, d = len(input_data), len(input_data[0])
        x = 3
        y = 3
        learning_rate = 0.1
        epochs = 500
        kohosom = msom.SOM(x=x, y=y, num_dim=d, learning_rate=learning_rate, random_seed=42)
        kohosom.train(input_data, epochs, verbose=False)
        weights = kohosom.weights
        expected = np.array([[[0.28159951, 0.05420668, 0.76096138],
            [0.63371375, 0.11178439, 0.4893027 ],
            [0.79768856, 0.4304459,  0.08398071]],

            [[0.8631365,  0.37828383, 0.69850119],
            [0.63083062, 0.40398228, 0.54066131],
            [0.4689983,  0.36391014, 0.1581002 ]],

            [[0.73989457, 0.94655753, 0.69165429],
            [0.48920864, 0.64787048, 0.86439795],
            [0.3045666,  0.45381696, 0.31682791]]])
        assert_array_almost_equal(weights, expected)

        log(green("PASS"), test_name, "")

def main():
    """
    This is the main function that executes the tests for the SOM algorithm.

    It creates an instance of the TestSOM class and runs various test methods
    to evaluate the SOM implementation.

    The tests include:
    - Setting up the test environment
    - Testing the Euclidean distance calculation
    - Testing the Gaussian function
    - Testing the quantization error calculation
    - Testing the decay function
    - Training the SOM
    - Predicting using the trained SOM

    If any exception occurs during the execution of the tests, the traceback is logged.

    Returns:
    None
    """
    test = TestSOM()

    try:
        test.setUp()
        test.test_euclidean_distance()
        # test.test_gaussian()
        test.test_quantization_error()

        test.test_decay_function()
        test.test_train()
        test.test_predict()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()
