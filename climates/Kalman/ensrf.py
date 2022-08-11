
from scipy.linalg import sqrtm
from dask import array as da
import numpy as np
from sympy import divisors


def ensrf(xbm, xb, y, H, obs_error, client, chunk_size, localization=None, P=None):
    """Ensemble Square Root Filter

    Args:
        xbm (dask.Array): Ensemble mean shape (Nx,)
        xb (dask.Array): Ensemble members shape (Nx,Ne)
        y (dask.Array): Observation vector shape (Ny,)
        H (dask.Array): Forward Matrix shape (Ny,Nx)
        client (dask.Client): Client 
        localization (dask.Array, optional): Localization matrix for schur product shape (Nx,Nx). Defaults to None.
        P (dask.Array, optional): Covariance matrix if precomputed. Defaults to None.
        obs_error (float): variance term of observation errors

    Returns:
        xam: analysis of the mean shape (Nx,)
        xap: analysis of difference about the mean shape (Nx,Ne)
        xa: analysis of the state vector shape (Nx,Ne)

    Note: at each intermediate step we persist and scatter the quantities to have smaller tasks by the end of the steps
    """

    "Get quantities and load them to distributed memory"

    "Mean"
    xbm = xbm.persist()
    xbm = client.scatter(xbm, broadcast=True).result()

    "State vector"
    xb = xb.persist()
    xb = client.scatter(xb, broadcast=True).result()

    "Difference about the mean"
    xbp = xb - xbm  # Get x_prime
    xbp = xbp.persist()  # load to memory
    # scatter to multiple workers for easier computations
    xbp = client.scatter(xbp, broadcast=True).result()

    Nx = xbp.shape[0]
    Ne = xbp.shape[1]

    "Observations"
    y = y[~np.isnan(y)]  # get rid of NaNs
    Ny = y.shape[0]  # get dimension of observational space
    y = y.persist()
    y = client.scatter(y, broadcast=True).result()

    "Forward Matrix"
    H = client.persist(H)
    H = client.scatter(H, broadcast=True).result()

    "Begin calculating quantities:\
        1) Hxm: dot product between H and xbm used for updating the ensemble mean \
        2) Hxp: dot product between H and xbp used to update the ensemble members"

    assert (H.shape[1] == xbm.shape[0],
            f'Shapes are not compatible for dot product: \
            H has shape {H.shape} and xbm has shape {xbm.shape}')

    Hxm = da.matmul(H, xbm)
    Hxm = client.persist(Hxm)
    Hxm = client.scatter(Hxm, broadcast=True).result()

    assert (H.shape[1] == xbp.shape[0],
            f'Shapes are not compatible for dot product: \
            H has shape {H.shape} and xbp has shape {xbp.shape}')

    Hxp = da.matmul(H, xbp)
    Hxp = client.persist(Hxp)
    Hxp = client.scatter(Hxp, broadcast=True).result()

    "The following is optional"
    if P is not None:
        "Here we compute all the relevant quantities including P (covariance matrix) \
            1) PHT, shape (Nx,Ny)\
            2) HPHT, shape (Ny,Ny) \
        "

        P = client.persist(P)
        P = client.scatter(P, broadcast=True).result()
        assert (H.shape[1] == P.shape[0],
                f'Shapes are not compatible for dot product: \
            P has shape {P.shape} and H.T has shape {H.T.shape}')

        "Calculate PHT"
        PHT = da.matmul(P, H.T)
        PHT = client.persist(PHT)
        PHT = client.scatter(PHT, broadcast=True).result()

        "Calculate HPHT"

        HPHT = da.matmul(H, PHT)
        HPHT = client.persist(HPHT)
        HPHT = client.scatter(HPHT, broadcast=True).result()

    else:

        "Compute PHT and HPHT"

        PHT = da.matmul(xbp, Hxp.T) / (Ne-1)
        PHT = client.persist(PHT)
        PHT = client.scatter(PHT, broadcast=True).result()

        HPHT = da.matmul(Hxp, Hxp.T) / (Ne-1)
        HPHT = client.persist(HPHT)
        HPHT = client.scatter(HPHT, broadcast=True).result()

    if localization is not None:
        "Here we want to localize the covariance matrix P in the quantities PHT and HPHT.\
         Let C be the localization matrix of shape (Nx,Nx). Then to localize PHT and HPHT we need to compute \
            1) CHT, shape (Nx,Ny) \
            2) HCHT, shape (Ny,Ny) \
        "

        "CHT"

        CHT = da.matmul(localization, H.T)
        CHT = client.persist(CHT)
        CHT = client.scatter(CHT, broadcast=True).result()

        PHT = da.multiply(CHT, PHT)  # localized PHT
        PHT = client.persist(PHT)
        PHT = client.scatter(PHT, broadcast=True).result()

        HCHT = da.matmul(H, CHT)
        HCHT = client.persist(HCHT)
        HCHT = client.scatter(HCHT, broadcast=True).result()

        HPHT = da.multiply(HCHT, HPHT)  # localized HPHT
        HPHT = client.persist(HPHT)
        HPHT = client.scatter(HPHT, broadcast=True).result()

    else:
        PHT = PHT
        HPHT = HPHT

    R = obs_error * da.eye(Ny)  # get matrix of observation error
    R = client.persist(R)
    R = client.scatter(R, broadcast=True).result()

    "In the following we compute the quantities of interest for the Kalman Gain and the Adjusted Kalman Gain (K_tilde)"

    "Kalman Gain K = PHT(HPHT + R)⁻¹"

    # first compute HPHT + R

    HPHTR = HPHT + R
    # Now invert HPHT + R
    # Note that to invert a matrix chunks must be all squares
    # We define first the list of divisors of Ny

    # rechunk the matrix to invert
    HPHTR = HPHTR.rechunk((chunk_size, chunk_size))
    HPHTR = client.persist(HPHTR)
    HPHTR = client.scatter(HPHTR, broadcast=True).result()

    HPHTR_sqrt = da.linalg.cholesky(HPHTR)
    HPHTR_sqrt = client.persist(HPHTR_sqrt)
    HPHTR_sqrt = client.scatter(HPHTR_sqrt, broadcast=True).result()

    HPHTR_sqrt_inv = da.linalg.inv(HPHTR_sqrt)
    HPHTR_sqrt_inv = client.persist(HPHTR_sqrt_inv)
    HPHTR_sqrt_inv = client.scatter(HPHTR_sqrt_inv, broadcast=True).result()

    # invert the matrix
    HPHTR_inv = da.matmul(HPHTR_sqrt_inv, HPHTR_sqrt_inv.T)

    HPHTR_inv = client.persist(HPHTR_inv)
    HPHTR_inv = client.scatter(HPHTR_inv, broadcast=True).result()

    "Compute the Kalman Gain K:"

    K = da.matmul(PHT, HPHTR_inv)
    K = client.persist(K)
    K = client.scatter(K, broadcast=True).result()

    "Compute the mean analysis"
    innovation = y - Hxm  # innovation factor
    innovation = innovation.persist()
    innovation = client.scatter(innovation, broadcast=True).result()

    xam = xbm + da.matmul(K, innovation)  # ensemble mean analysis

    "Compute the quantities of interest for the Adjusted Kalman Gain (K_tilde)"

    R = R.rechunk((chunk_size, chunk_size))
    R = client.persist(R)
    R = client.scatter(R,
                       broadcast=True).result()
    R_sqrt = da.linalg.cholesky(R)
    R_sqrt = client.persist(R_sqrt)
    R_sqrt = client.scatter(R_sqrt, broadcast=True).result()

    "Now we need to invert the quantities sqrt(HPHT + R) and sqrt(HPHT) + sqrt(R)"
    "Note that also here we need to have squared chunks"

    to_inv = HPHTR_sqrt + R_sqrt
    to_inv = to_inv.rechunk((chunk_size, chunk_size))
    to_inv = client.persist(to_inv)
    to_inv = client.scatter(to_inv, broadcast=True).result()

    inv = da.linalg.inv(to_inv)
    inv = client.persist(inv)
    inv = client.scatter(inv, broadcast=True).result()

    "Now we compute the Adjusted Kalman Gain"

    to_multiply = da.matmul(HPHTR_sqrt_inv.T, inv)
    to_multiply = client.persist(to_multiply)
    to_multiply = client.scatter(to_multiply, broadcast=True).result()

    K_tilde = da.matmul(PHT, to_multiply)
    K_tilde = client.persist(K_tilde)
    K_tilde = client.scatter(K_tilde, broadcast=True).result()

    KHxp = da.matmul(K_tilde, Hxp)
    KHxp = client.persist(KHxp)
    KHxp = client.scatter(KHxp, broadcast=True).result()

    xap = xbp - KHxp
    xap = xap.persist()
    xa = xap + xam
    xa = xa.persist()
    return xa, xap, xam
