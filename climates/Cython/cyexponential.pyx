from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport exp as c_exp
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _exponential(double theta, double alpha, double nu, double tau) nogil:
    cdef double s, t, angle, lhs, rhs, max_term

    angle = theta/alpha
    lhs = 1-tau**2
    rhs = c_exp(-(angle)**(nu))
    max_term = lhs * rhs
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def exponential(float[:, ::1] theta, int alpha, double nu, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _exponential(theta[i,j],alpha, nu, tau )
    return np.asarray(weights)