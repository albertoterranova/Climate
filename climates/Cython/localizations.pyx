from libc.math cimport sqrt, sin, cos, fabs, atan2
from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _c2w(double theta, int alpha, double tau) nogil:
    cdef double s, lhs, rhs, max_term

    lhs = 1 + tau*alpha*theta
    rhs = 1 - alpha*theta
    max_term = lhs*rhs**tau
    s = max(0.0, max_term) 
    return s

@cython.wraparound(False)
@cython.boundscheck(False)
def c2w(double[:, ::1] theta, int alpha, double tau):
    cdef int M = theta.shape[0]
    cdef int N = theta.shape[1]
    cdef int i, j            
    cdef double[:,::1] weights = np.zeros((M,N),dtype=np.double)
    cdef double s
    for i in prange(M,nogil=True):
        for j in range(N):
            weights[i,j] += _c2w(theta[i,j],alpha, tau)
    return np.asarray(weights)



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