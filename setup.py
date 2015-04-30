#!/usr/bin/env python

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(name="Fooling_Larson",
          version=__version__,
          description="Larson Intrinsic Representation System",
          url="https://github.com/shaunrong/Fooling-Larson",
          long_description=open(os.path.join(module_dir, 'README.md')),
          author="Ziqin (Shaun) Rong, Manuel Cabral",
          author_email="rongzq08@gmail.com, cabman567@gmail.com",
          license="MIT License",
          packages=find_packages(),
          zip_safe=False,
          install_requires=["numpy", "pyyaml"],
          classifiers=["Development Status :: 2 - Pre-Alpha",
                       "Topic :: Scientific/Engineering :: Artificial Intelligence"]
          )
