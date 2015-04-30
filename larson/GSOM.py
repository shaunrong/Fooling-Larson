#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np


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
        self.alpha = alpha
        self.lam = lam
        self.tou = tou
        self.n = n
        self.iter = 0

        # Making sure parameters have valid values
        #TODO: fix the types
        if type(self.alpha) != float:
            raise TypeError('alpha must be a float')
        if self.alpha < 0 or 1 < self.alpha:
            raise ValueError('alpha must be between 0 and 1. Received: ' + str(self.alpha))

        if type(self.lam) != int:
            raise TypeError('lam must be an int')
        if self.lam < 1:
            raise ValueError('lam must be an integer greater than or equal to 1. Received: ' + str(self.lam))

        if type(self.n) != int:
            raise TypeError('n must be an int')
        if self.n < 1:
            raise ValueError('n must be an integer greater than or equal to 1. Received: . Received: ' + str(self.n))
        
        #initiating the map
        #TODO: normalize each vector
        self.map = np.random.rand(2,2,n)

    def update(self, feature):
        """
        update the map with a input vector (either array or numpy array)
        """
        def __get_closest_match():
            """
            Retrieves the indices of the cell with the smallest Euclidean distance to feature
            """
            errors = np.zeros(self.map.shape[:2])
            #TODO: Refactor so its not two nested for-loops
            for x in xrange(self.map.shape[0]):
                for y in xrange(self.map.shape[1]):
                    errors[x][y] = np.linalg.norm(self.map[x][y] - feature)

            return np.unravel_index( np.argmin(errors), errors.shape)

        def __neighborhood_of(cell):
            """
            Retrieves the cells closest to the specified cell.
            Includes horizontal, vertical and diagonal cells.
            """
            return [(cell[0] + x, cell[1] + y)
                    for x in xrange(-1,2)
                    for y in xrange(-1,2)
                    if (x != 0 or y != 0)
                    and (0 <= cell[0] + x < self.map.shape[0])
                    and (0 <= cell[1] + y < self.map.shape[1])]

        def __update_cell(x, y):
            """
            Updates the cell according to the following formula:
            m_i(t+1) = m_i(t) + alpha(t) * ( x(t) - m_i(t) )
            """
            if not (0 <= x <= self.map.shape[0]) or not(0 <= y <= self.map.shape[1]):
                raise ValueError('Invalid cell ('+ str(x) +',' + str(y) + ') in map of shape' + str(self.map.shape[:2]) )

            self.map[x][y] = (1-self.alpha)*self.map[x][y] + self.alpha*feature

        #Check that input is valid
        if type(feature) != list and type(feature) != np.ndarray:
            raise TypeError('Input feature vector has to be a list or numpy array.')
        if len(feature) != self.n:
            raise ValueError('Input feature has the wrong dimensionality. Got ' +str(len(feature)) + ', Expected ' + str(self.n))

        #Update cells
        match = __get_closest_match()
        for cell in __neighborhood_of(match):
            __update_cell(*cell)

        #Grow map if lam iterations have passed
        self.iter += 1
        if self.iter >= self.lam:
            self.grow()
            self.iter = self.iter % self.lam

    def grow(self):
        """
        grow the map every lam iteration
        """
        if self.iter % self.lam == 0:
            pass
            #TODO: update the map

    def check(self):
        """
        check if the map has converge
        :return: boolean
        """
        pass
        #TODO: check the map's convergence