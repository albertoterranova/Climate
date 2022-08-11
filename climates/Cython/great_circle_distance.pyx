from libc.math cimport sqrt, sin, cos, fabs, atan2
import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def great_circle_distance_matrix(float[:, ::1] X, float[:, ::1] Y):
    cdef int M = X.shape[0]
    cdef int K = X.shape[1]
    cdef int N = Y.shape[0]
    cdef int i, j

    cdef float[:, ::1] D = np.empty((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            D[i, j] = _great_circle_distance(X[i, :], Y[j, :])
    return np.asarray(D)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _great_circle_distance(float[::1] x, float[::1] y):
    cdef float lat1, lat2, lon1, lon2, dlon, dlat, num, den
    lat1, lon1 = x[0], x[1]
    lat2, lon2 = y[0], y[1]

    dlon = fabs(lon1 - lon2)
    dlat = fabs(lat1 - lat2)
    
    num = sqrt(
        (cos(lat2)*sin(dlon))**2 +
        ((cos(lat1)*sin(lat2)) - (sin(lat1)*cos(lat2)*cos(dlon)))**2)

    den = (
        (sin(lat1)*sin(lat2)) +
        (cos(lat1)*cos(lat2)*cos(dlon)))

    return atan2(num, den)


