#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np
from itertools import product

class GSOM(object):

    def __init__(self, alpha, lam, tou, n=7):
        """
        Class for Growth Self-Organize Map.
        :param alpha: learning rate
        :param lam: every lam iterations, there is a growth phase
        :param tou: parameter controlling converging
        :param n: dimension of the feature vector at each cell of the SOM
        :return: A initiated 2 * 2 SOM
        """
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
            raise ValueError('n must be an integer greater than or equal to 1. Received: . Received: ' + str(self.n))
        
        #initiating the map
        self._map = np.random.rand(2, 2, n)
        self._mapping = np.array([[0,1],[2,3]])
        self._context = {}

    @property
    #TODO: pick better name, this currently overrides the map function
    def map(self):
        if not self.converged():
            print "Please be noted, the GSOM training hasn't converged."
        return self._map

    @property
    def converged(self):
        return self._converged

    def __neighborhood_of(self,x,y):
        """
        Retrieves the cells closest to the specified cell.
        Only includes vertical and horizontal cells.
        """
        return [(x + i, y + j)
                for i, j in product(xrange(-1, 2), xrange(-1, 2))
                if ((i + j) % 2 != 0)
                and (0 <= x + i < self._map.shape[0])
                and (0 <= y + j < self._map.shape[1])]

    def __get_closest_match(self,feature):
        """
        Retrieves the indices of the cell with the smallest Euclidean distance to feature
        """
        errors = np.zeros(self._map.shape[:2])
        #TODO: Refactor so its not two nested for-loops
        for x,y in product(xrange(self._map.shape[0]), xrange(self._map.shape[1])):
            errors[x][y] = np.linalg.norm(self._map[x][y] - feature)

        return np.unravel_index( np.argmin(errors), errors.shape)

    def __update_cell(self,feature, x, y):
        """
        Updates the cell according to the following formula:
        m_i(t+1) = m_i(t) + alpha(t) * ( x(t) - m_i(t) )
        """
        if not (0 <= x <= self._map.shape[0]) or not(0 <= y <= self._map.shape[1]):
            raise ValueError('Invalid cell ('+ str(x) +',' + str(y) + ') in map of shape' + str(self._map.shape[:2]))

        self._map[x][y] = (1-self._alpha)*self._map[x][y] + self._alpha*feature

    def update(self, feature):
        """
        update the map with a input vector (either array or numpy array)
        """

        #Check that input is valid
        if type(feature) != list and type(feature) != np.ndarray:
            raise TypeError('Input feature vector has to be a list or numpy array.')
        if len(feature) != self._n:
            raise ValueError('Input feature has the wrong dimensionality. Got ' +str(len(feature)) + ', Expected '
                             + str(self._n))

        if self._converged == True:
            return

        #Update cells
        match = self.__get_closest_match(feature)
        for cell in self.__neighborhood_of(*match):
            self.__update_cell(feature, *cell)

        #Map feature to this cell
        if self._mapping.shape <= match:
            raise ValueError("match is out of range, looking for cell " + str(match) + " in mapping of shape " + str(self._mapping.shape))
        if self._context.has_key(self._mapping[match]):
            self._context[self._mapping[match]].append(feature)
        else:
            self._context[self._mapping[match]] = [feature]

        #Grow map if lam iterations have passed
        self._iter += 1
        if self._iter >= self._lam:
            self.grow()
            self._iter = self._iter % self._lam

    def __get_highest_error_cell(self):
        """
        It gets the cell e with the highest quantization error defined as:
        qe_i = sum_(x_j) || m_i - x_j ||
        where the x_j's are the features mapped to that cell.
        """
        raise NotImplementedError()

    def __get_worst_neighbor(self, cell):
        """
        Find the most dissimilar neighbor of a cell.
        """
        c_vector = self._map[cell[0]][cell[1]]
        return max(self.__neighborhood_of(*cell),
                   key = lambda n: np.linalg.norm( c_vector - self._map[n[0]][n[1]] ) )

    def __grow_map(self,c1,c2):
        """
        Grows the map between the two specified cells
        """
        raise NotImplementedError()

    def grow(self):
        """
        Grow the map. It's expected that this is called by GSOM.update
        """
        most_error_cell = self.__get_highest_error_cell()
        worst_neighbor = self.__get_worst_neighbor(most_error_cell)
        self.__grow_map(most_error_cell, worst_neighbor)

    def check(self):
        """
        check if the map has converge
        :return: boolean
        """
        pass
        #TODO: check the map's convergence, if converged, change self._converged = True
