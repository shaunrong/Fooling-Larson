#!/usr/bin/env python

__author__ = 'Manuel Cabral, Shaun (Ziqin) Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun (Ziqin) Rong'
__email__ = 'cabman567@gmail.com, rongzq08@gmail.com'

import numpy as np
from itertools import product


class GSOM(object):

    def __init__(self, alpha, lam, tou, n=7, size=10):
        """
        Class for Growth Self-Organize Map.
        :param alpha: learning rate
        :param lam: every lam iterations, there is a growth phase
        :param tou: parameter controlling converging
        :param n: dimension of the feature vector at each cell of the SOM
        :param size: size of map is (size x size)
        :return: A initiated 2 * 2 SOM
        """
        #TODO: Make alpha decrease with number of iterations
        self._alpha = alpha
        self._lam = lam
        self._tou = tou
        self._n = n
        self._iter = 0
        self._converged = False

        # Making sure parameters have valid values
        #TODO: fix the types
        if type(self._alpha) != float:
            raise TypeError('alpha must be a float')
        if self._alpha < 0 or 1 < self._alpha:
            raise ValueError('alpha must be between 0 and 1. Received: ' + str(self._alpha))

        if type(self._lam) != int:
            raise TypeError('lam must be an int')
        if self._lam < 1:
            raise ValueError('lam must be an integer greater than or equal to 1. Received: ' + str(self._lam))

        if type(self._n) != int:
            raise TypeError('n must be an int')
        if self._n < 1:
            raise ValueError('n must be an integer greater than or equal to 1. Received: . Received: ' + str(self._n))

        #initiating the map
        self._map = np.random.rand(size, size, n)
        shape = self._map.shape[:2]
        self._mapping = np.arange(np.prod(shape), dtype=int).reshape(shape)
        self._context = {}

        #total quantization error variables
        self._qe_u = np.random.rand(n)
        self._input_vectors = []

    @property
    def map(self):
        if not self.converged:
            print "Please note, the GSOM training hasn't converged."
        return self._map

    @property
    def converged(self):
        return self._converged

    def __neighborhood_of(self, x, y, distance="1"):
        """
        Retrieves the cells closest to the specified cell.
        Includes vertical, horizontal, diagonal cells.

        Distance is used to specify to distance from specified cell.
        It's the l2 norm squared from (x,y).
        """
        return [(x + i, y + j)
                for i, j in product(xrange(-1, 2), xrange(-1, 2))
                if (0 < i**2 + j**2 <= distance)
                and (0 <= x + i < self._map.shape[0])
                and (0 <= y + j < self._map.shape[1])]

    def __get_closest_match(self, feature):
        """
        Retrieves the indices of the cell with the smallest Euclidean distance to feature
        """
        errors = np.zeros(self._map.shape[:2])
        #TODO: Refactor so its not two nested for-loops
        for x, y in product(xrange(self._map.shape[0]), xrange(self._map.shape[1])):
            errors[x][y] = np.linalg.norm(self._map[x][y] - feature)

        return np.unravel_index(np.argmin(errors), errors.shape)

    def __update_cell(self, feature, x, y):
        """
        Updates the cell according to the following formula:
        m_i(t+1) = m_i(t) + alpha(t) * ( x(t) - m_i(t) )
        """
        if not (0 <= x <= self._map.shape[0]) or not(0 <= y <= self._map.shape[1]):
            raise ValueError('Invalid cell (' + str(x) + ',' + str(y) + ') in map of shape' + str(self._map.shape[:2]))

        self._map[x][y] = (1 - self._alpha) * self._map[x][y] + self._alpha * feature

    def update(self, feature):
        """
        update the map with a input vector (either array or numpy array)
        """

        #Check that input is valid
        if type(feature) != list and type(feature) != np.ndarray:
            raise TypeError('Input feature vector has to be a list or numpy array.')
        if len(feature) != self._n:
            raise ValueError('Input feature has the wrong dimensionality. Got ' + str(len(feature)) + ', Expected '
                             + str(self._n))

        if self._converged:
            print "GSOM already converged."
            return

        #We need to make sure that it's a numpy array
        feature = np.array(feature)

        #Update variables for mean quantization error
        self._qe_u = (1 - self._alpha) * self._qe_u + self._alpha * feature
        self._input_vectors.append(feature)

        #Update cells
        match = self.__get_closest_match(feature)
        for cell in self.__neighborhood_of(*match,distance=2):
            self.__update_cell(feature, *cell)
        self.__update_cell(feature, *match)

        #Map feature to this cell
        if self._mapping.shape <= match:
            raise ValueError("match is out of range, looking for cell " + str(match) + " in mapping of shape " + str(self._mapping.shape))

        if self._context.has_key(self._mapping[match]):
            self._context[self._mapping[match]].append(feature)
        else:
            self._context[self._mapping[match]] = [feature]

        self.check()

    def __quantization_error_of(self, cell):
        """
        Computes the following for the cell:
        qe_i = sum_(x_j) || m_i - x_j ||
        where the x_j's are the features mapped to that cell.
        """
        #If there aren't any mappings to this cell, return 0
        if not self._context.has_key(self._mapping[cell]):
            return 0

        input_vectors = self._context[self._mapping[cell]]
        quantized_error = lambda inp: np.linalg.norm(self._map[cell] - inp)
        return sum([ quantized_error(inp) for inp in input_vectors])/len(input_vectors)

    def __total_quantization_error_of(self):
        """
        The total quantization error is a measurement of the variability of the data.
        It can be thought of as if all the input vectors went through one cell.
        """
        return sum([np.linalg.norm(self._qe_u - inp) for inp in self._input_vectors])

    def __mean_quantization_error_of(self):
        """
        This is average of the quantization errors of all the cells.
        """
        quantization_errors = [self.__quantization_error_of(cell)
                               for cell in product(xrange(self._map.shape[0]),xrange(self._map.shape[1]))
                               if self._context.has_key(self._mapping[cell])]
        return sum(quantization_errors)#/len(quantization_errors)

    def check(self):
        """
        check if the map has converge
        :return: boolean
        """
        if (self.__mean_quantization_error_of() < self._tou*self.__total_quantization_error_of()):
            self._converged = True
            print "Map converged after", self._iter, "iterations. Shape is", self._map.shape[:2]
