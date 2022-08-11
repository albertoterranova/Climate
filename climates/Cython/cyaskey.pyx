from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport exp as c_exp
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _askey(double theta, int alpha) nogil:
    cdef double s, angle, max_term

    angle = theta*alpha
    max_term = 1-angle
    s = max(0.0, max_term**2) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def askey(double[:, ::1] theta, int alpha):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _askey(theta[i,j],alpha)
    return np.asarray(weights)