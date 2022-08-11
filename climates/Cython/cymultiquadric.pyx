from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport cos as c_cos
import cython
from cython.parallel import prange,parallel
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _multiquadric(double theta, double alpha, double nu, double tau) nogil:
    cdef double s, t, num, num_base, num_exp, den, den_base, max_term
    num_base = 1 - tau
    num_exp = 2*alpha
    num = num_base**num_exp
    den_base = 1 + tau**2 - 2*tau*c_cos(theta)
    den = den_base**alpha
    max_term = num / den
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def multiquadric(float[:, ::1] theta, int alpha, double nu, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _multiquadric(theta[i,j],alpha, nu, tau )
    return np.asarray(weights)