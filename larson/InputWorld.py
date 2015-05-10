#!/usr/bin/env python
import random
import yaml
import numpy as np

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


class Digits(object):
    def __init__(self, seg, norm):
        """
        This class generates a input world of digital numbers, using number <-> vector info from
        digits.yaml (norm by human)

        :param seg: How many segments each edge of the digital numbers are cut into
        :param norm: file path to the digits.yaml (norm from human setting)
        The picture below specifies the position each edge inside the input vector of digits.yaml.

         [0]
         --
    [1] |  |  [2]
    [3]  --
    [4] |  |  [5]
    [6]  --
        """
        self._seg = seg
        self._dim = seg * 7
        self._norm = norm
        self._symMap = self._get_sym_map()

    @property
    def dim(self):
        """
        :return: return the dimensionality of the input vector to Larson sys
        """
        return self._dim

    @property
    def sym_map(self):
        """
        :return: the symbol map for each
        """
        return self._symMap

    def ran_input_unsup(self):
        """
        Provide a randomized input for Larson sys unsupervised learning phase. Randomization works like:
        0 in input will be randomized to 0.05~0.1
        1 in input will be randomized to 0.9~0.95
        :return: an input vector
        """
        vec = random.choice(self.sym_map.values())
        ran_input = []
        for dim in vec:
            if dim == 0:
                ran_input.append(random.random() * 0.05 + 0.05)
            if dim == 1:
                ran_input.append(0.95 - random.random() * 0.05)
        return ran_input

    def ran_input_sup(self):
        """
        Provide a randomized input for Larson supervised learning phase. Randomization works like:
        0 in input will be randomized to 0~0.15
        1 in input will be randomized to 0.85~1.00
        :return: an input vector
        """
        vec = random.choice(self.sym_map.values())
        ran_input = []
        for dim in vec:
            if dim == 0:
                ran_input.append(random.random() * 0.15)
            if dim == 1:
                ran_input.append(1 - random.random() * 0.15)
        return ran_input

    def _get_sym_map(self):
        """
        this function reads the norm .yaml file and produce a Symbol Map corresponding the segments each edge is
        cut into. e.g. if seg = 2, one -> [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        :return: a dictionary similar to norm .yaml file, but with number of segments
        """
        with open(self._norm, 'r') as digits:
            sym_map = yaml.load(digits)
        for key, value in sym_map.iteritems():
            new_map = np.array([], dtype=int)
            for element in value:
                if element == 0:
                    new_map = np.append(new_map, np.zeros(self._seg, dtype=int))
                if element == 1:
                    new_map = np.append(new_map, np.ones(self._seg, dtype=int))
            sym_map[key] = new_map
        return sym_map