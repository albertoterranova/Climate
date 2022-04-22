import numpy as np
import dask.array as da
from numpy.polynomial.legendre import legval


# Covariance functions are taken from
# "Isotropic covariance functions on spheres: Some properties and modeling considerations, Guinnes J., Fuentes M. 2016 "
# The great circle distance formula (Vincenty) takes a standard formulation easily recoverable from any website/book/lecture notes e.g. Wikipedia: Vincenty's Formulae
# The client.submit() trick allows us taking a future object, do the required computations and returning another future object
# Using only one cluster as in this toy experiment this removes the need for further unpacking, e.g. future.result() or dask.compute(), and useless speed loss.
# To bring consistency to the paper above, we use the following notation to document operations:
#       -) x^y stands for x to the power of y
#       -) x_+ stands for max(0,x)
#       -) we use da.array(k), for k any number, so that dask understands how to broadcast
#       -) sometimes we use da.true_divide because it allows for float division
#       -) sometimes we want to use da.float_power because it allows for float power
#  ###


def gcd_dask_future(lat, lon, client):
    """great circle distance using Vincenty's Formulae

    Args:
        lat (1-d dask array): latitudinal point$
        lon (1-d dask array): longitudinal points
        client 

    Returns:
        dask array great circle distance matrix of all pairwise points
    """

    r = 6371
    l, L = da.meshgrid(lat, lon)
    l, L = client.submit(da.Array.flatten, l), client.submit(
        da.Array.flatten, L)  # get glatten arrays of longitude and latitudes
    # get array of all coordinates pair 2xN
    sources = client.submit(da.vstack, (l, L))
    sources = client.submit(da.transpose, sources)  # transpose -> Nx2
    sources_rad = client.submit(da.radians, sources)  # get radians
    sources_rad = sources_rad.result().persist()  # save into memory for computations
    # difference of all point coordinates
    delta_lambda = sources_rad[:, [1]] - sources_rad[:, 1]
    phi1 = sources_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = sources_rad[:, 0]  # (1 x M) array of destination latitudes

    # we want to use the Vincenty's formula for the unit sphere and then multiply by the radius of the earth
    # Δσ = arctan2(sqrt((cos(ϕ1) * sin(Δλ))² +  (cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ))²),cos(ϕ1) * cos(ϕ2) * cos(Δλ))

    cos_phi2 = client.submit(da.cos, phi2)  # cos(ϕ2)
    cos_phi1 = client.submit(da.cos, phi1)  # cos(ϕ2)
    cos_d_l = client.submit(da.cos, delta_lambda)  # cos(Δλ)
    sin_d_l = client.submit(da.sin, delta_lambda)  # sin(Δλ)
    sin_phi1 = client.submit(da.sin, phi1)  # sin(ϕ1)
    sin_phi2 = client.submit(da.cos, phi2)  # cos(ϕ2)
    a = client.submit(da.multiply, cos_phi2, sin_d_l)  # cos(ϕ1) * sin(Δλ)
    b = client.submit(da.power, a, 2)  # (cos(ϕ1) * sin(Δλ))²

    c = client.submit(da.multiply, cos_phi1, sin_phi2)  # cos(ϕ1) * sin(ϕ1)
    d = client.submit(da.multiply, cos_phi2, sin_phi1)  # cos(ϕ2) * sin(ϕ1)
    e = client.submit(da.multiply, d, cos_d_l)  # cos(ϕ2) * sin(ϕ1) * cos(Δλ)
    # cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ)
    f = client.submit(da.subtract, c, e)
    # (cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ))²
    g = client.submit(da.power, f, 2)
    # (cos(ϕ1) * sin(Δλ))² +  (cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ))²
    h = client.submit(da.add, b, g)
    # sqrt((cos(ϕ1) * sin(Δλ))² +  (cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ))²)
    i = client.submit(da.sqrt, h)

    k = client.submit(da.multiply, sin_phi1, sin_phi2)  # sin(ϕ1) * sin(ϕ2)
    l = client.submit(da.multiply, cos_phi1, cos_phi2)  # cos(ϕ1) * cos(ϕ2)
    m = client.submit(da.multiply, l, cos_d_l)  # cos(ϕ1) * cos(ϕ2) * cos(Δλ)

    # arctan2(sqrt((cos(ϕ1) * sin(Δλ))² +  (cos(ϕ1) * sin(ϕ1) - cos(ϕ2) * sin(ϕ1) * cos(Δλ))²),cos(ϕ1) * cos(ϕ2) * cos(Δλ))
    x = client.submit(da.arctan2, 1, m)
    y = client.submit(da.multiply, x, r)  # get distance in km
    return y

def bernoulli(alpha, theta, lower,n, upper, client):
    theta = client.scatter(theta)  
    theta = client.persist(theta.result()) 
    lower_index = da.arange(lower, -1, 1)
    upper_index = da.arange(1, upper+1, 1)
    coefs_lower = da.zeros_like(theta)
    coefs_upper = da.zeros_like(theta)
    for k in lower_index:
        coefs_lower += da.float_power(abs(k),(-2*n)) * da.exp(k*theta*1j)
    for k in upper_index:
        coefs_upper += da.float_power(abs(k), (-2*n)) * da.exp(k*theta*1j)
    lower = client.scatter(coefs_lower)
    upper = client.scatter(coefs_upper)
    return client.submit(da.add,lower,upper)

def circular_matern(alpha, theta, nu, lower, upper, client):
    theta = client.scatter(theta)  
    theta = client.persist(theta.result())  
    index = da.arange(lower, upper + 1, 1)
    coefs = da.zeros_like(theta)
    for k in index:
        coefs += (alpha**2 + k**2)**(-nu - .5)*da.exp(k*theta*1j)
    
    coefficients = client.scatter(coefs)
    return coefficients


def legendre_matern(sigma, alpha, nu, theta, N, client):
    """ ψ(θ)=σ² * Σ_{k=0}^N (α² + k²)^{-ν - .5} * P_k(cos(θ))
    Args:
        sigma (float): variance
        alpha (float/int): range, α>0
        nu (float): smoothness, ν>0
        theta (array): great-circle distance
        N (int): truncation order
        client 

    Returns:
        covariance matrix of Legendre-Matern type: array
    """
    cos_theta = client.scatter(da.cos(theta)) # scatter to multiple workes cos(θ)
    cos_theta = client.gather(cos_theta) # gather result
    cos_theta = cos_theta.persist() # load into memory for calculations
    term = da.zeros_like(theta) # coefficients [c_0,c_1,...,c_N] needed for sum: c_0*P_0(cos(θ)) + ... + c_N*P_N(cos(θ))
    coefs = np.zeros(N)
    for k in range(N):
        coefs[k] = (alpha**2 + k**2)**(-nu - .5) 
        ###
        # c_0 = (α² + 0²)^(-ν - 0.5) 
        # c_1 = (α² + 1²)^(-ν - 0.5)
        # .
        # .
        # .
        # c_N = (α²2 + N²)^(-ν - 0.5)   
        ###
    legendre = client.submit(legval, cos_theta, coefs) 
    ###
    # legendre = Σ_{k=0} ^ N (α² + k²)^{-ν - .5} * P_k(cos(θ))
    #            = (α² + 0²)^(-ν - 0.5)*P_0(cos(θ)) + ... + (α² + N²)^(-ν - 0.5)*P_n(cos(θ))
    #            = c_0*P_0(cos(θ)) + ... + c_N*P_N(cos(θ))
    #            = Σ_{k=0} ^ N c_k * P_k(cos(θ)) done by legvan(cos_theta,coefs)
    ###
    return client.submit(da.multiply, sigma**2, legendre) ## just multiply legendre by sigma²



def powered_exp(alpha, theta, tau, client):  # exp{-(αθ)^τ}
    """Powered Exponential covariance matrix

    Args:
        alpha(int/float): range in km, α>0
        theta(future): great circle distance matrix
        tau(float): in (0,1]
        client
    Returns:
        future dask array
    """
    a = client.submit(da.multiply, theta, alpha)  # αθ
    b = client.submit(da.float_power, a, tau)  # (αθ)^τ
    c = client.submit(da.multiply, b, -1)  # -(αθ)^τ
    f = client.submit(da.exp, c)  # exp{-(αθ)^τ}
    return f


def spherical_dask(theta, alpha, client):  # (1+αθ/2)(1-αθ)²_+
    """Spherical covariance matrix 

    Args:
        theta (future): great circle distance matrix
        alpha (int): range, α>0
        client

    Returns:
        future dask array
    """
    # make the mask for places farer than alpha km
    theta = client.submit(da.ma.masked_greater, theta, alpha)
    a = client.submit(da.multiply, theta, da.array(alpha))  # αθ
    b = client.submit(da.subtract, da.array(1), a)  # (1-αθ)
    c = client.submit(da.power, b, 2)  # (1-αθ)²
    g = client.submit(da.maximum, da.array(0), c)  # (1-αθ)²_+
    d = client.submit(da.true_divide, a, 2)  # αθ/2
    e = client.submit(da.add, da.array(1), d)  # (1+αθ/2)
    f = client.submit(da.multiply, g, e)  # (1+αθ/2)(1-αθ)²_+
    # set covariance to location far apart to 0
    f = client.submit(da.ma.filled, f, 0)
    return f


def cauchy_dask(alpha, theta, nu, tau, client):  # (1 + (αθ)^ν)^−(τ/ν)
    """Cauchy Covariance matrix 

    Args:
        alpha (int):  α>0
        theta (future): great circel distance matrix 
        nu (float): in (0,1]
        tau (float): τ>0
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.multiply, da.array(alpha), theta)  # αθ
    b = client.submit(da.float_power, a, nu)  # (αθ)^ν
    c = client.submit(da.add, da.array(1), b)  # 1 + (αθ)^ν
    const = -tau/nu  # −(τ/ν)
    d = client.submit(da.power, c, const)  # (1 + (αθ)^ν)^−(τ/ν)
    return d


def multiquadric_dask(alpha, theta, tau, client):  # (1 − τ)^2α /(1 + τ² − 2τ cos(θ))^α
    """Multiquadric covariance matrix 

    Args:
        alpha (int): α>0
        theta (future): great circle distance matrix 
        tau (float): τ in (0,1)
        client 
    Returns:
        future dask array
    """
    a = (1 - tau)**(2*alpha)  # (1 − τ)^2α
    b = 1 + tau**2  # 1 + τ²
    c = 2*tau  # 2τ
    d = client.submit(da.cos, theta)  # cos(θ)
    e = client.submit(da.multiply, c, d)  # 2τ cos(θ)
    f = client.submit(da.add, b, e)  # 1 + τ² − 2τ cos(θ)
    g = client.submit(da.power, f, alpha)  # (1 + τ² − 2τ cos(θ))^α
    # (1 − τ)^2α /(1 + τ² − 2τ cos(θ))^α
    h = client.submit(da.true_divide, a, g)
    return h


def sine_dask(theta, nu, client):  # 1-(sin(θ/2))^ν
    """sine power covariance matrix 

    Args:
        theta (future): great circle distance matrix 
        nu (float): ν in (0,2)
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.true_divide, theta, 2)  # θ/2
    b = client.submit(da.sin, a)  # sin(θ/2)
    c = client.submit(da.power, b, nu)  # (sin(θ/2))^ν
    d = client.submit(da.subtract, da.array(1), c)  # 1-(sin(θ/2))^ν
    return d


def askey_dask(theta, alpha, tau, client):  # ((1 − αθ)^τ)_+
    """Askey Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (int): α>0
        tau (float): τ>2
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.multiply, da.array(alpha), theta)  # αθ
    b = client.submit(da.subtract, 1, a)  # 1 − αθ
    c = client.submit(da.power, b, tau)  # (1 − αθ)^τ
    d = client.submit(da.maximum, da.array(0), c)  # ((1 − αθ)^τ)_+
    return d


def c2_dask(theta, alpha, tau, client):  # (1 + ταθ)((1 − αθ)^τ))_+
    """C2-Wendland Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (int): α>1/π
        tau (float): τ>4
        client 

    Returns:
        future dask array
    """
    a = tau*alpha  # τα
    b = client.submit(da.multiply, a, theta)  # ταθ
    c = client.submit(da.add, da.array(1), b)  # 1 + ταθ

    d = client.submit(da.multiply, da.array(alpha), theta)  # αθ
    e = client.submit(da.subtract, da.array(1), d)  # 1 − αθ
    f = client.submit(da.float_power, e, tau)  # (1 − αθ)^τ)
    g = client.submit(da.maximum, 0, f)  # ((1 − αθ)^τ))_+
    h = client.submit(da.multiply, c, g)  # (1 + ταθ)((1 − αθ)^τ))_+
    return h


def c4_dask(theta, alpha, tau, client):  # (1 + ταθ + (τ² - 1)/3 * (αθ)² )((1 − αθ)^τ))_+
    """C4-Wendland Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (float): α>1/π
        tau (float): τ>6
        client 

    Returns:
        future dask array
    """
    a = tau*alpha  # τα
    b = client.submit(da.multiply, a, theta)  # ταθ
    c = client.submit(da.add, da.array(1), b)  # 1 + ταθ
    i = (tau**2 - 1)/3  # (τ² - 1)/3
    d = client.submit(da.multiply, da.array(alpha), theta)  # αθ
    l = client.submit(da.power, d, 2)  # (αθ)²
    m = client.submit(da.multiply, i, l)  # (τ² - 1)/3 * (αθ)²
    n = client.submit(da.add, c, m)  # 1 + ταθ + (τ² - 1)/3 * (αθ)²

    e = client.submit(da.subtract, da.array(1), d)  # 1 − αθ
    f = client.submit(da.float_power, e, tau)  # ((1 − αθ)^τ))_+
    g = client.submit(da.maximum, 0, f)  # ((1 − αθ)^τ))_+
    # (1 + ταθ + (τ² - 1)/3 * (αθ)² )((1 − αθ)^τ))_+
    h = client.submit(da.multiply, c, g)

    return h


def poisson_dask(theta, alpha, client):  # (1-α²)/((1-2α*cos(θ) + α²)^(3/2))
    """Poisson Covariance Matrix

    Args:
        theta (future): great circle distance matrix 
        alpha (float): α in (0,1)
        client 

    Returns:
        future dask array
    """

    a = 1 - alpha**2  # (1-α²)
    b = client.submit(da.cos, theta)  # cos(θ)
    c = client.submit(da.multiply, da.array(2*alpha), b)  # 2α*cos(θ)
    d = client.submit(da.subtract, da.array(1+alpha**2), c)  # 1-2α*cos(θ) + α²
    e = client.submit(da.float_power, d, 1.5)  # (1-2α*cos(θ) + α²)^(3/2)

    # (1-α²)/((1-2α*cos(θ) + α²)^(3/2))
    f = client.submit(da.true_divide, a, e)

    return f


# 1 - (Σ(xam - xref)²)/(Σ(xbm - xref)²) where the sum runs from 1 to 6 (months)
def re_score_dask(xam, xr, xbm, client):
    """_summary_

    Args:
        xam (future): updated ensemble members mean
        xr (future): reference anomaly dataset
        xbm (future): non-updated ensemble members mean
        client 

    Returns:
        h: future dask array of pointwise RE score
        i: future dask array of spacetime average
    """
    a = client.submit(da.subtract, xam, xr)  # xam - xref
    b = client.submit(da.power, a, 2)  # (xam - xref)²
    c = client.submit(da.sum, b, axis=0)  # Σ(xam - xref)²

    d = client.submit(da.subtract, xbm, xr)  # xbm - xref
    e = client.submit(da.power, d, 2)  # (xbm - xref)²
    f = client.submit(da.sum, e, axis=0)  # Σ(xbm - xref)²

    g = client.submit(da.true_divide, c, f)  # Σ(xam - xref)²/(Σ(xbm - xref)²

    # 1 - (Σ(xam - xref)²)/(Σ(xbm - xref)²)
    h = client.submit(da.subtract, da.array(1), g)

    # takes the space-time mean excluding the NaN's
    i = client.submit(da.nanmean, h)
    return h, i
