#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


class Digits(object):
    def __init__(self, seg):
        """
        This class generates a input world of digital numbers, using number <-> vector info from digits.yaml
        :param seg: How many segments each edge of the digital numbers are cut into

        The picture below specifies the position each edge inside the input vector of digits.yaml.

         [0]
         --
    [1] |  |  [2]
    [3]  --
    [4] |  |  [5]
    [6]  --
        """
        self.seg = seg
