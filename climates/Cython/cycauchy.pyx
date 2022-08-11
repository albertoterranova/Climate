from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport exp as c_exp
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _cauchy(double theta, double alpha, double nu, double tau) nogil:
    cdef double s, t, angle, base, power, max_term

    angle = theta*alpha
    base = 1+(alpha*theta)**(nu)
    power = - tau/nu
    max_term = base**power
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def cauchy(float[:, ::1] theta, int alpha, double nu, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _cauchy(theta[i,j],alpha, nu, tau )
    return np.asarray(weights)