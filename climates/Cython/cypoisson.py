# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:53:35 2022

@author: alberto
"""
from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

EXTENSIONS = [
    Extension('cypoisson',
              ['cypoisson.pyx'],
              extra_compile_args=['/openmp', '/O2', '/fp:fast'],
              ),
]

setup(
    name='cypoisson',
    version='0.1',
    packages=['cypoisson'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
)


