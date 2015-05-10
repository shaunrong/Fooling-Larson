#!/usr/bin/env python
from larson.Clustering import UPGMA

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from larson.InputWorld import Digits
from larson.GSOM import GSOM


def main():
    input_world = Digits(1, norm='../norms/digits.yaml')
    SOM = GSOM(alpha=0.1, lam=10, tou=0.1)
    #debug
    it = 0
    while not SOM.converged:
        SOM.update(input_world.ran_input_unsup())
        print it
        it += 1
    clusters = UPGMA(SOM)
    clusters.clustering()
    print clusters.representative
    print clusters.association

if __name__ == '__main__':
    main()
