#!/usr/bin/env python
import yaml

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np
import larson.GSOM as GSOM


class UPGMA(object):
    def __init__(self, SOM, clusters=10):
        """
        This class uses the Pair Group Method with Arithmetic Mean (UPGMA) algorithm to cluster SOM Map.
        :param GSOM: input SOM world (a numpy array)
        :param clusters: number of clusters the UPGMA clusters into
        """
        if type(SOM) != GSOM.GSOM:
            raise TypeError("SOM input has to be a GSOM object.")
        self._world = GSOM
        self._clusters = clusters

        self._association = {}
        self._representative = {}
        self._left_cells = []
        shape = SOM.map.shape[0:2]
        for i in range(np.prod(shape)):
            self._association[i] = [[int(i / SOM.map.shape[0]), i % SOM.map.shape[0]]]
            self._representative[i] = SOM.map[np.unravel_index(i, shape)]
            self._left_cells.append(i)

        self._resemblance = {}

        for i in range(len(self._representative)):
            for j in range(i):
                self._resemblance[(i, j)] = np.linalg.norm(self._representative[i] - self._representative[j])

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
        while len(self._left_cells) != self._clusters:
            min_index = min(self._resemblance, key=self._resemblance.get)
            self._update_clusters(min_index)

    def _update_clusters(self, min_index):
        cluster_i = max(min_index)
        cluster_j = min(min_index)

        self._association[cluster_j].extend(self._association[cluster_i])
        del self._association[cluster_i]

        self._representative[cluster_j] = (self._representative[cluster_j] + self._representative[cluster_i]) / 2.0
        del self._representative[cluster_i]

        self._left_cells.remove(cluster_i)
        self._left_cells.sort()

        for i in self._left_cells:
            if i != cluster_j:
                self._resemblance[UPGMA._reorder_tuple(cluster_j, i)] = np.linalg.norm(self._representative[i] -
                                                                                      self._representative[cluster_j])
        for key in self._resemblance.keys():
            if cluster_i in key:
                del self._resemblance[key]

    @staticmethod
    def _reorder_tuple(a, b):
        if a == b:
            raise ValueError("There is no diagonal term UPGMA._resemblance.")
        if a < b:
            return tuple([b, a])
        if a > b:
            return tuple([a, b])


class KMeans(object):
    def __init__(self, SOM, clusters = 10):
        if type(SOM) != GSOM.GSOM:
            raise TypeError("SOM input has to be a GSOM object.")
        self._world = GSOM
        self._clusters = clusters
        self._representative = {}
        self._association = {}
        with open('../norms/digits.yaml', 'r') as digits:
            sym_map = yaml.load(digits)
        for key, value in sym_map.iteritems():
            self._representative[key] = value
            self._association[key] = np.array([])

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
        updated_representative = {}
        for cell in np.nditer(self._world):
            distance = {}
            for key, value in self._representative.iteritems():
                distance[key] = np.linalg.norm(cell - value)
            belong_group = min(distance, key=distance.get)
            self._association[belong_group].append(cell)
        for key in self._representative.keys():
            updated_representative[key] = np.mean(self._association[key], axis=0)

        while self._representative != updated_representative:
            self._representative = updated_representative
            for cell in np.nditer(self._world):
                distance = {}
                for key, value in self._representative.iteritems():
                    distance[key] = np.linalg.norm(cell - value)
                belong_group = min(distance, key=distance.get)
                self._association[belong_group].append(cell)
            for key in self._representative.keys():
                updated_representative[key] = np.mean(self._association[key], axis=0)