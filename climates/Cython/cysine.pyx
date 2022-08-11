from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport sin as c_sin
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _sine(double theta, double alpha, double nu) nogil:
    cdef double s, max_term

    max_term = 1 - c_sin(theta/2)**nu
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def sine(float[:, ::1] theta, int alpha, double nu, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _sine(theta[i,j],alpha, nu )
    return np.asarray(weights)