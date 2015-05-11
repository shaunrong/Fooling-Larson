#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import unittest
from larson.GSOM import GSOM
from larson.Clustering import UPGMA
import numpy as np


class TestClustering(unittest.TestCase):
    def test_on_evenly_distributed_GSOM(self):
        SOM = GSOM(alpha=0.1, lam=10, tou=0.1)
        SOM._converged = True
        SOM._map = np.ones((20, 20, 7))
        for i in range(10):
            for j in range(10):
                SOM._map[i][j] = np.ones(7) * 10
            for j in range(10, 20):
                SOM._map[i][j] = np.ones(7) * 20
        for i in range(10, 20):
            for j in range(10):
                SOM._map[i][j] = np.ones(7) * 30
            for j in range(10, 20):
                SOM._map[i][j] = np.ones(7) * 40
        clusters = UPGMA(SOM, clusters=4)
        clusters.clustering()

        for key in clusters.association.keys():
            self.assertEqual(len(clusters.association[key]), 100)

if __name__ == '__main__':
    unittest.main()