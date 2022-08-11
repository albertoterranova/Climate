from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

EXTENSIONS = [
    Extension('cycauchy',
              ['cycauchy.pyx'],
              extra_compile_args=['/openmp', '/O2', '/fp:fast'],
              ),
]

setup(
    name='cauchy',
    version='0.1',
    packages=['cauchy'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
)
