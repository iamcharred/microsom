import numpy as np
import time
import pickle
import somutils
from numpy import (unravel_index, linalg, random,
                   power, exp, zeros, arange, meshgrid)
from collections import defaultdict
from numpy.linalg import norm

class SOM:
    def __init__(self, x, y, num_dim, learning_rate = 0.1, sigma=None,
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
            learning_rate : initial learning rate, optional (default=0.1)
                (at the iteration t we have
                learning_rate(t) = learning_rate / (1 + t/T)
                where T is #num_iteration/2)
            sigma : float, optional (default=None)
                Spread of the neighborhood function, needs to be adequate
                to the dimensions of the map.
                (at the iteration t we have sigma(t) = sigma / (1 + t/T)
                where T is #num_iteration/2)
            random_seed : int, optional (default=None)
                Random seed to use.
            """
        self._x = x
        self._y = y
        self._num_dim = num_dim

        self._learning_rate = learning_rate
        if sigma is None:
            self._sigma = np.maximum(self._x, self._y)/2.0
        else:
            self._sigma = sigma

        if random_seed is None:
            # Seed generator with system time
            self._random_generator = random.RandomState(int(time.time()))
        else:
            self._random_generator = random.RandomState(random_seed)

        self._weights = self._random_generator.random((self._x, self._y, self._num_dim)).astype('float64')
        #self._weights = self._random_generator.rand(self._x, self._y, self._num_dim).astype('float64')*2-1
        #self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        # initialize the distance map
        self._distance_map = zeros((x, y))

        # initialize grid for neighborhood function
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

    @property
    def weights(self):
        """Returns the weights of the map."""
        return self._weights

    @property
    def sigma(self):
        """Returns the initial neighborhood radius."""
        return self._sigma

    @property
    def learning_rate(self):
        """Returns the initial learning rate."""
        return self._learning_rate

    @property
    def map_size(self):
        """Returns the size of the som map."""
        return self._x, self._y

    @property
    def num_dim(self):
        """Returns the dimension of the input vectors and weights."""
        return self._num_dim

    @property
    def distance_map(self):
        """Returns the distance map of the weights."""
        return self._distance_map


    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix


    def _asymptotic_decay(self, param, t):
        """Decay function of the learning process.
        Parameters
        ----------
        param : float
            parameter to decay.
        t : int
            current iteration.
        """

        return param * np.exp(-t / self._lambda)


    # def _euclidean_distance(self, current_input_vector, w):
    #     """Returns the squared euclidean distance between two vectors.

    #     Parameters
    #     ----------
    #     current_input_vector : np.array
    #         Current input vector to use for training.
    #     """

    #     return linalg.norm(subtract(current_input_vector, w), axis=-1)


    def _compute_distance_map(self, current_input_vector):
        """Updates matrix distance map, in this matrix
           the element i,j is the distance of the neuron i,j to the current_input vector.

        Parameters
        ----------
        current_input_vector : np.array
            Current input vector to use for training.
        """
        self._distance_map = linalg.norm(current_input_vector - self._weights, axis=-1)
        # for i in range(self._x):
        #     for j in range(self._y):
        #         self._distance_map[i,j] = self._euclidean_distance(current_input_vector, self._weights[i,j])


    def winner(self, current_input_vector, verbose=False):
        """Get coordinates of winning neuron for the sample.

        Parameters
        ----------
        current_input_vector : np.array
            Current input vector to use for training.
        """

        self._compute_distance_map(current_input_vector)

        if verbose:
            print("distance map: {}".format(self._distance_map))
            print("min distance: {}".format(self._distance_map.argmin()))
            print("coord of node with min distance: {}".format(unravel_index(self._distance_map.argmin(),self._distance_map.shape)))

        return unravel_index(self._distance_map.argmin(),
                             self._distance_map.shape)


    def _update(self, current_input_vector, win, t, verbose=False):
        """Updates the weights of the neurons in the amp.

        Parameters
        ----------
        current_input_vector : np.array
            Current input vector to use for training.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        """

        # decrease learning rate
        alpha = self._asymptotic_decay(self._learning_rate, t) #influence
        # decrease neighborhood radius
        sig = self._asymptotic_decay(self._sigma, t) #neigborhood radius

        # generate neighborhood matrix centered on winner
        # alpha * neighborhood_function
        g = self._gaussian(win, sig)*alpha

        # w_new = w_t + aplha * gaussian_neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, current_input_vector-self._weights)
        #self._weights += np.einsum('ij, ijk->ijk', g, np.subtract(np.tile(current_input_vector, (self._x, self._y, 1) ),self._weights))
        # self._weights += g[:, :, np.newaxis] * (current_input_vector - self._weights)


    def train(self, data, num_iteration, verbose=False):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix to train the SOM.

        num_iteration : int
            The weights will be updated len(data)*num_iteration times.
        """
        data_len = len(data)
        assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal! {0} , {1}".format(self._num_dim, len(data[0]))

        iterations = somutils.build_iteration_indexes(data_len, num_iteration,
                                              self._random_generator)

        self._lambda = num_iteration / np.log(self._sigma)

        for i, current_input_index in enumerate(iterations):
            t = i // data_len
            if verbose:
                print("t: {}, training sample: {}".format(t, current_input_index))
            self._update(data[current_input_index], self.winner(data[current_input_index], verbose=False), i, verbose=True)


    def win_map(self, data, return_indices=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""

        # assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal!"
        assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal! {0} , {1}".format(self._num_dim, len(data[0]))

        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.winner(x)].append(i if return_indices else x)
        return winmap

    def predict(self, data):
        """Get coordinates of the winning neuron for the input data.

        Parameters
        ----------
        data : np.array
            Input data.
        """

        # assert len(data) == self._num_dim, "Data dimension and input dimension must be equal!"
        assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal! {0} , {1}".format(self._num_dim, len(data[0]))

        return self.winner(data)

    def pickle_model(self, filename):
        """Save the SOM object to a file.

        Parameters
        ----------
        filename : text
            File path and filename where the model will be saved.
        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""

        # assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal!"
        assert len(data[0]) == self._num_dim, "Data dimension and input dimension must be equal! {0} , {1}".format(self._num_dim, len(data[0]))

        error = 0
        for i in range(len(data)):
            error += norm(data[i] - self._weights[self.winner(data[i])])
        error /= len(data)
        return error