#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from larson.InputWorld import Digits
from larson.GSOM import GSOM
from larson.Clustering import UPGMA


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

    @property
    def results(self):
        return self._results

    def train(self):

