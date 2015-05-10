#!/usr/bin/env python
from larson.Clustering import UPGMA
from larson.GSOM import GSOM
from larson.InputWorld import Digits
from larson.ClusterAssociation import TrainDigits

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def main():
    #Generate input world
    input_world = Digits(1, norm='../norms/digits.yaml')

    #Unsupervised learning phase
    #Self-organizing map
    SOM = GSOM(alpha=0.1, lam=10, tou=0.1)
    while not SOM.converged:
        SOM.update(input_world.ran_input_unsup())
    #Clustering
    clusters = UPGMA(SOM)
    clusters.clustering()

    #Unsupervised Learning phase
    sup_train = TrainDigits(input_world, clusters)
    sup_train.train()

    #Fooling it

if __name__ == '__main__':
    main()