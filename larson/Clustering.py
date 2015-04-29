#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np
from GSOM import GSOM


class UPGMA(object):
    def __init__(self, GSOM, clusters=10):
        """
        This class uses the Pair Group Method with Arithmetic Mean (UPGMA) algorithm to cluster SOM Map.
        :param GSOM: input SOM world (a numpy array)
        :param clusters: number of clusters the UPGMA clusters into
        """
        if type(GSOM) != GSOM:
            raise TypeError("SOM input has to be a GSOM object.")
        self._world = GSOM
        self._clusters = clusters

        self._association = {}  # {'cluster1' : [indexes of cells below to that cluster]}
        for i in range(clusters):
            self._association['cluster' + str(i)] = []

        self._representative = {}
        for i in range(clusters):
            self._representative['cluster' + str(i)] = []

    @property
    def association(self):
        return self._association

    @property
    def representative(self):
        return self._representative

    def clustering(self):
        """
        Class the GSOM world into the right number of clusters, update cluster association dictionary as well as the
        representative of each cluster.
        """
        #TODO: Implement the UPGMA clustering process.
        pass