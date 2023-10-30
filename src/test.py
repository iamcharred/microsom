import sys
import utils
import unittest
import traceback
import numpy as np
import microsom as msom

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.linalg import norm
from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal)


def green(s):
        return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)

class TestSOM(unittest.TestCase):

    def setUp(self):
        test_name = "setUp"
        self.som = msom.SOM(4, 4, 3, sigma=0.5, learning_rate=0.5, random_seed=42)
        self.som._weights = ones((4, 4, 3))  # fake weights
        self.som._weights[1, 0] = ([0,0,0])
        self._input_data = np.array([[1,1,1],[2,2,2],[3,3,3]])
        log(green("PASS"), test_name, "")

    def test_euclidean_distance(self):
        test_name = "test_euclidean_distance"
        self.som._compute_distance_map(self._input_data[0])
        d = self.som.distance_map
        assert_almost_equal(d[1,0], 1.73205080)
        log(green("PASS"), test_name, "")

    def test_gaussian(self):
        test_name = "test_gaussian"
        bell = self.som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)
        log(green("PASS"), test_name, "")

    def test_predict(self):
        test_name = "test_predict"
        assert self.som.predict(self._input_data[1]) == self.som.winner(self._input_data[1])
        log(green("PASS"), test_name, "")

    # def test_win_map(self):
    #         winners = self.som.win_map([[5.0], [2.0]])
    #         assert winners[(2, 3)][0] == [5.0]
    #         assert winners[(1, 1)][0] == [2.0]

    # def test_win_map_indices(self):
    #         winners = self.som.win_map([[5.0], [2.0]], return_indices=True)
    #         assert winners[(2, 3)] == [0]
    #         assert winners[(1, 1)] == [1]

    def test_quantization_error(self):
            assert_almost_equal(self.som.quantization_error(self._input_data), 1.73205080)

    def test_decay_function(self):
        test_name = "test_decay_function"
        self.som.train(self._input_data, 100)
        assert self.som._asymptotic_decay(1,1) == 1 * np.exp(-1 / (100 / np.log(.5)))
        assert self.som._asymptotic_decay(1,2) == 1 * np.exp(-2 / (100 / np.log(.5)))
        log(green("PASS"), test_name, "")

    def test_train(self):
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
        weights = kohosom._weights
        expected = np.array([[[0.55794153, 0.73421595, 0.90004395],
                                [0.68701643, 0.95383317, 0.61168179],
                                [0.83248862, 0.93113686, 0.49046788]],

                                [[0.87398804, 0.31393875, 0.7332091 ],
                                [0.82115085, 0.50176916, 0.61215266],
                                [0.85617872, 0.57743056, 0.07606036]],

                                [[0.28124754, 0.05404332, 0.76038402],
                                [0.63175815, 0.11261231, 0.48901839],
                                [0.53855108, 0.34825438, 0.16172409]]])
        assert_array_almost_equal(weights, expected)
        log(green("PASS"), test_name, "")

def main():
    test = TestSOM()

    try:
        test.setUp()
        test.test_euclidean_distance()
        # test.test_gaussian()
        test.test_quantization_error()

        test.test_decay_function()
        test.test_predict()
        test.test_train()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()