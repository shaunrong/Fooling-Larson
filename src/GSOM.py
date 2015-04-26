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
        #TODO: initiate the map

    def update(self, feature):
        """
        update the map with a input vector (either array or numpy array)
        """
        if type(feature) != list and type(feature) != np.ndarray:
            raise TypeError('Input feature vector has to be a list or numpy array.')
        if len(feature) != self.n:
            raise ValueError('Input feature has the wrong dimensionality.')

        self.iter += 1
        pass
        #TODO: update the map

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
