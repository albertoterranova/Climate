# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:52:24 2022

@author: alberto
"""

from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

EXTENSIONS = [
    Extension('localizations',
              ['localizations.pyx'],
              extra_compile_args=['/openmp', '/O2', '/fp:fast'],
              ),
]

setup(
    name='localizations',
    version='0.1',
    packages=['localizations'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
)
