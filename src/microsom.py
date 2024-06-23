import time
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from numpy import (unravel_index, linalg, zeros, arange, meshgrid)

import somutils

class SOM:
    """Kohonen Self Organizing Maps implementation.

    This class represents a Kohonen Self Organizing Maps (SOM) model.
    It can be used for training and predicting on high-dimensional data.

    Attributes
    ----------
    weights : numpy.ndarray
        The weights of the map.
    sigma : float
        The initial neighborhood radius.
    learning_rate : float
        The initial learning rate.
    map_size : tuple
        The size of the SOM map. (x, y)
    num_dim : int
        The dimension of the input vectors and weights.
    distance_map : numpy.ndarray
        The distance map of the weights.

    Methods
    -------
    train(data, num_iteration, verbose=False)
        Trains the SOM model.
    predict(data)
        Get coordinates of the winning neuron for the input data.
    pickle_model(filename)
        Save the SOM object to a file.
    quantization_error(data)
        Returns the quantization error computed as the average distance between each input sample and its best matching unit.
    """

    def __init__(self, x, y, num_dim, learning_rate=0.1, sigma=None,
                random_seed=None) -> None:
        """Initializes a Kohonen Self Organizing Maps.

        Parameters
        ----------
        x : int
            x dimension of map
        y : int
            y dimension of map
        num_dim : int
            Number of the elements of the vectors in input.
        learning_rate : float, optional (default=0.1)
            Initial learning rate. At each iteration t, the learning rate is calculated as learning_rate(t) = learning_rate / (1 + t/T),
            where T is num_iteration/2.
        sigma : float, optional (default=None)
            Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
            At each iteration t, the neighborhood radius is calculated as sigma(t) = sigma / (1 + t/T),
            where T is num_iteration/2.
        random_seed : int, optional (default=None)
            Random seed to use.
        """
        self._x = x
        self._y = y
        self._num_dim = num_dim

        self._learning_rate = learning_rate
        if sigma is None:
            self._sigma = np.maximum(self._x, self._y) / 2.0
        else:
            self._sigma = sigma

        if random_seed is None:
            # Seed generator with system time
            self._random_generator = np.random.default_rng(int(time.time()))
        else:
            self._random_generator = np.random.default_rng(random_seed)

        self._weights = self._random_generator.random((self._x, self._y, self._num_dim)).astype('float64')

        """initialize the distance map"""
        self._distance_map = zeros((x, y))

        """initialize grid for neighborhood function"""
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

    @property
    def weights(self) -> np.ndarray:
        """Returns:
            numpy.ndarray: The weights of the map.
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights: np.ndarray) -> None:
        """Sets the weights of the map.

        Args:
            new_weights (np.ndarray): The new weights to be set for the map.
        """
        self._weights = new_weights

    @property
    def sigma(self) -> float:
        """Returns:
            float: The initial neighborhood radius.
        """
        return self._sigma

    @property
    def learning_rate(self) -> float:
        """Returns
            float: the initial learning rate."""
        return self._learning_rate

    @property
    def map_size(self) -> tuple:
        """Returns:
            tuple: A tuple containing the width and height of the som map.
        """
        return self._x, self._y

    @property
    def num_dim(self) -> int:
        """Returns:
            int: The dimension of the input vectors and weights.
        """
        return self._num_dim

    @property
    def distance_map(self) -> np.ndarray:
        """Returns:
            np.ndarray: The distance map of the weights.
        """
        return self._distance_map

    def _gaussian(self, c: int, sigma: float) -> np.ndarray:
        """
        Returns a Gaussian matrix centered at index c.

        Parameters:
            c (int): The index representing the center of the Gaussian.
            sigma (float): The standard deviation of the Gaussian.

        Returns:
            np.ndarray: A matrix representing the Gaussian distribution.
        """
        d = 2 * sigma ** 2
        ax = np.exp(-np.power(self._xx - self._xx.T[c], 2) / d)
        ay = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T  # the external product gives a matrix

    def _asymptotic_decay(self, param: float, t: int) -> float:
        """Decay function of the learning process.

        Parameters
        ----------
        param : float
            Parameter to decay.
        t : int
            Current iteration.

        Returns
        -------
        float
            The decayed parameter value.
        """
        return param * np.exp(-t / self._lambda)

    def _compute_distance_map(self, current_input_vector: np.ndarray) -> None:
        """Updates matrix distance map.

        In this matrix, the element i,j is the distance of the neuron i,j to the current_input vector.

        Parameters
        ----------
        current_input_vector : np.ndarray
            Current input vector to use for training.
        """
        self._distance_map = linalg.norm(current_input_vector - self._weights, axis=-1)

    def winner(self, current_input_vector: np.ndarray, verbose: bool = False) -> tuple:
        """Get coordinates of winning neuron for the sample.

        Parameters
        ----------
        current_input_vector : np.ndarray
            Current input vector to use for training.
        verbose : bool, optional
            If True, print additional information, by default False.

        Returns
        -------
        tuple
            The coordinates (i,j) of the winning neuron.
        """
        self._compute_distance_map(current_input_vector)

        if verbose:
            print(f"distance map: {self._distance_map}")
            print(f"min distance: {self._distance_map.argmin()}")
            print(f"coord of node with min distance: {unravel_index(self._distance_map.argmin(), self._distance_map.shape)}")

        return unravel_index(self._distance_map.argmin(),
                             self._distance_map.shape)

    def _update(self, current_input_vector: np.ndarray, win: tuple, t: int, verbose: bool = False) -> None:
        """Updates the weights of the neurons in the map.

        Parameters
        ----------
        current_input_vector : np.ndarray
            Current input vector to use for training.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        verbose : bool, optional
            If True, print additional information, by default False.
        """
        # decrease learning rate
        alpha = self._asymptotic_decay(self._learning_rate, t)  # influence
        # decrease neighborhood radius
        sig = self._asymptotic_decay(self._sigma, t)  # neighborhood radius

        # generate neighborhood matrix centered on winner
        # alpha * neighborhood_function
        g = self._gaussian(win, sig) * alpha

        self._weights += np.einsum('ij, ijk->ijk', g, current_input_vector - self._weights)

    def train(self, data: np.ndarray, num_iteration: int, verbose: bool = False) -> None:
        """Trains the SOM.

        Parameters
        ----------
        data : np.ndarray
            Data matrix to train the SOM.
        num_iteration : int
            The weights will be updated len(data)*num_iteration times.
        verbose : bool, optional
            If True, print additional information during training, by default False.
        """
        self._input_data = data
        data_len = len(data)
        assert len(data[0]) == self._num_dim, f"Data dimension and input dimension must be equal! {self._num_dim}, {len(data[0])}"

        iterations = somutils.build_iteration_indexes(data_len, num_iteration,
                                                      self._random_generator)

        self._lambda = num_iteration / np.log(self._sigma)

        for i, current_input_index in enumerate(iterations):
            t = i // data_len
            if verbose:
                print(f"t: {t}, training sample: {current_input_index}")
            self._update(data[current_input_index], self.winner(data[current_input_index], verbose=False), i, verbose=True)

    def win_map(self, data: np.ndarray, return_indices: bool = False) -> dict:
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True

        Parameters
        ----------
        data : np.ndarray or list
            Data matrix to map on the SOM.
        return_indices : bool, optional (default=False)
            If True, returns the indices of the elements instead of the elements themselves.

        Returns
        -------
        dict
            A dictionary where the keys are the coordinates (i,j) and the values are lists of patterns or indices.
        """
        assert len(data[0]) == self._num_dim, f"Data dimension and input dimension must be equal! {self._num_dim}, {len(data[0])}"

        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.winner(x)].append(i if return_indices else x)
        return winmap

    def predict(self, data: np.ndarray) -> tuple:
        """Get coordinates of the winning neuron for the input data.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        tuple
            The coordinates (i,j) of the winning neuron.
        """
        # print(f"Data: {data}, Input dimension: {self._num_dim}")
        # assert len(np.array(data)) == self._num_dim, f"Data dimension and input dimension must be equal! {self._num_dim}, {len(np.array(data))}"

        return self.winner(data)

    def pickle_model(self, filename: str) -> None:
        """Save the SOM object to a file.

        Parameters
        ----------
        filename : str
            File path and filename where the model will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def quantization_error(self, data: np.ndarray) -> float:
        """Returns the quantization error computed as the average distance between each input sample and its best matching unit.

        Parameters
        ----------
        data : np.ndarray
            Data matrix to compute the quantization error.

        Returns
        -------
        float
            The quantization error.
        """
        assert len(data[0]) == self._num_dim, f"Data dimension and input dimension must be equal! {self._num_dim}, {len(data[0])}"

        error = 0
        for i, datum in enumerate(data):
            error += np.linalg.norm(datum - self._weights[self.winner(datum)])
        error /= len(data)
        return error

    def plot_input(self) -> None:
        """Plots the input data of the SOM as a grid of subplots."""
        plt.imshow(np.array(self._input_data))

    def plot_weights(self) -> None:
        """Plots the weights of the SOM as a grid of subplots."""
        plt.imshow(self._weights)