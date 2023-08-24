#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

extension = Extension(name="mass_matrix_func",
                      sources=["mass_matrix_func.pyx",
                               "mass_matrix_func_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="mass_matrix_func",
      ext_modules=cythonize([extension], language_level="3str"))