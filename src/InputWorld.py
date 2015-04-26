#!/usr/bin/env python

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

    def ran_input(self):
        """
        Provide a randomized input for Larson sys. Randomization works like:
        0 in input will be randomized to 0~0.1
        1 in input will be randomized to 0.9~1
        :return: an input vector
        """
        return []

    def _get_sym_map(self):
        """
        this function reads the norm .yaml file and produce a Symbol Map corresponding the segments each edge is
        cut into. e.g. if seg = 2, one -> [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        :return: a dictionary similar to norm .yaml file, but with number of segments
        """
        #TODO: Calculate the sym_map
        return {}

