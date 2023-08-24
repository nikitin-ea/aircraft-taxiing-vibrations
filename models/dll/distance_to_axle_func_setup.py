#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

extension = Extension(name="distance_to_axle_func",
                      sources=["distance_to_axle_func.pyx",
                               "distance_to_axle_func_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="distance_to_axle_func",
      ext_modules=cythonize([extension], language_level="3str"))