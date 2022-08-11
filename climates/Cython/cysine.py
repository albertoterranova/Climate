from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

EXTENSIONS = [
    Extension('cysine',
              ['cysine.pyx'],
              extra_compile_args=['/openmp', '/O2', '/fp:fast'],
              ),
]

setup(
    name='sine',
    version='0.1',
    packages=['sine'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
)
