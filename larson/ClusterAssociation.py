#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from larson.InputWorld import Digits
from larson.GSOM import GSOM
from larson.Clustering import UPGMA
import numpy as np


class TrainDigits(object):
    def __init__(self, input_world, SOM, clusters):
        """
        This class does the supervised learning step of Larson system
        :param input_world: input world, a Digits object from larson.InputWorld module
        :param SOM: a trained growth self-organizing map
        :param clusters: results after applying clustering algo to the SOM.
        """
        if type(input_world) != Digits:
            raise TypeError('input_world is not a Digits object.')
        if type(SOM) != GSOM:
            raise TypeError('SOM is not a GSOM object.')
        if type(clusters) != UPGMA:
            raise TypeError('clusters is not a UPGMA object.')

        self._input_world = input_world
        self._SOM = SOM
        self._clusters = clusters
        self._results = {}

        self._association = {}
        for sym in self._input_world.sym_map.keys():
            self._results[sym] = {}
            self._association[sym] = {}
            for cluster in self._clusters.representative.keys():
                self._association[sym][cluster] = {'activated_num': 0, 'associated_inputs': []}

    @property
    def results(self):
        """
        only used after self.train() is called
        :return: results dictionary, with each sym of results[sym] has the following attributes:
            cluster_index: the index of cluster associated with this symbol;
            cluster_representative: the cluster representative vector of that cluster index from clustering results;
            guassian_mean: the mean of all input features associated with this cluster during supervised training phase,
                            notice it is input_world.dim dimensional vector
            guassian_std: the standard deviation of all input features associated with this cluster during supervised
                           training phase, notice it is input_world.dim dimensional vector
        """
        return self._results

    def train(self):
        for sym in self._input_world.sym_map.keys():
            self._results[sym] = {}
            for i in range(1000):
                self._associate(self._input_world.ran_input_sup(), sym)
        for sym in self._association.keys():
            associated_cluster = 0
            max_frequency = 0
            for cluster in self._association[sym].keys():
                if self._association[sym][cluster]['activated_num'] > max_frequency:
                    max_frequency = self._association[sym][cluster]['activated_num']
                    associated_cluster = cluster
            self._results[sym]['cluster_index'] = associated_cluster
            self._results[sym]['cluster_representative'] = self._clusters.representative[associated_cluster]
            self._results[sym]['guassian_mean'] = np.mean(
                np.array(self._association[sym][associated_cluster]['associated_inputs']), axis=0)
            self._results[sym]['guassian_std'] = np.std(
                np.array(self._association[sym][associated_cluster]['associated_inputs']), axis=0)

    def _associate(self, ran_input, sym):
        ran_input = np.array(ran_input)
        distance = {}
        for cluster in self._clusters.representative.keys():
            distance[cluster] = np.linalg.norm(ran_input - self._clusters.representative[cluster])
        associated_cluster = min(distance, key=distance.get)
        self._association[sym][associated_cluster]['activated_num'] += 1
        self._association[sym][associated_cluster]['associated_inputs'].append(ran_input)

