#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

extension = Extension(name="forcing_vector_func",
                      sources=["forcing_vector_func.pyx",
                               "forcing_vector_func_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="forcing_vector_func",
      ext_modules=cythonize([extension], language_level="3str"))