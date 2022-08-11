from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

EXTENSIONS = [
    Extension('cyaskey',
              ['cyaskey.pyx'],
              extra_compile_args=['/openmp', '/O2', '/fp:fast'],
              ),
]

setup(
    name='askey',
    version='0.1',
    packages=['askey'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
)
