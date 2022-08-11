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
cdef double _spherical(double theta, double alpha) nogil:
    cdef double s, t, angle, max_term, mult_term

    angle = theta/alpha
    max_term = (1 - angle)**2
    s = max(0.0, max_term) 
    mult_term = 1+(theta/(2*alpha))
    t = s * mult_term
    return t

@cython.wraparound(False)
@cython.boundscheck(False)
def spherical(float[:, ::1] theta, int alpha):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _spherical(theta[i,j],alpha)
    return np.asarray(weights)