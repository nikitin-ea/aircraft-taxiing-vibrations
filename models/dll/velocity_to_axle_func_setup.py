#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

extension = Extension(name="velocity_to_axle_func",
                      sources=["velocity_to_axle_func.pyx",
                               "velocity_to_axle_func_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="velocity_to_axle_func",
      ext_modules=cythonize([extension], language_level="3str"))