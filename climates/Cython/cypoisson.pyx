# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:06:21 2022

@author: alberto
"""

from libc.math cimport sqrt, sin, cos, fabs, atan2, exp
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float _poisson(float theta, float lambda0) nogil:
    
    cdef float num, den, p
    
    
    num = 1 - lambda0**2
    den = 1 - 2*lambda0* cos(theta) + lambda0**2
    p   = max(0.0 ,num / sqrt(den**3))
    
    return p
@cython.wraparound(False)
@cython.boundscheck(False)    
def poisson(float[:,::1] theta, float lambda0, float alpha):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j
    cdef float[:,::1] D = np.zeros((N,M),dtype=np.float32)
    cdef float num, den, p
    for i in prange(M,nogil=True):
        for j in range(N):
            D[i,j] += _poisson(theta[i,j],lambda0) 
    return np.asarray(D)
   
