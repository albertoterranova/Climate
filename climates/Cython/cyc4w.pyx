from matplotlib.pyplot import thetagrids
from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport exp as c_exp
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _c4w(double theta, int alpha, double tau) nogil:
    cdef double s, lhs, rhs, max_term
    first = tau*alpha*theta
    second = (tau**2 - 1)/3
    third = alpha*theta
    fourth = 1-alpha*theta
    max_term = first + second*third**2 * fourth**tau
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def c4w(double[:, ::1] theta, int alpha, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _c4w(theta[i,j],alpha, tau)
    return np.asarray(weights)