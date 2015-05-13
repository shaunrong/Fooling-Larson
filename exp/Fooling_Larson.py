#!/usr/bin/env python
from larson.Clustering import UPGMA
from larson.GSOMFixed import GSOM
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
    SOM = GSOM(alpha=0.1, lam=10, tou=0.01)
    while not SOM.converged:
        SOM.update(input_world.ran_input_unsup())
    #Clustering
    clusters = UPGMA(SOM)
    clusters.clustering()
    for key in clusters.association.keys():
        print "The number of cells in {} cluster is {}".format(key, len(clusters.association[key]))
    for key in clusters.representative.keys():
        print "The representative of {} in clusters are {}".format(key, clusters.representative[key])

    #Unsupervised Learning phase
    sup_train = TrainDigits(input_world, clusters)
    sup_train.train()

    #Fooling it
    for sym in sup_train.results.keys():
        for i in range(100):
            input_vec = input_world.ran_input_fooling_known(sym)
            match = sup_train.match(input_vec)
            if match != sym:
                print "Find a Mismatch, input vector is {}, matched sym is {}, the symbol should match is {}.".format(
                    input_vec, match, sym)

    for i in range(1000):
        input_vec = input_world.ran_input_fooling_not_known()
        match = sup_train.match(input_vec)
        if match != 'no_match':
            print "Find a Mismatch, input vector is {}, matched sym is {}, the symbol should match is {}.".format(
                input_vec, match, 'no_match')

if __name__ == '__main__':
    main()