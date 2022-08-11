import dask.array as da
from numpy.polynomial.legendre import legval


def bernoulli(alpha, theta, lower, n, upper, client):
    """Bernoulli covariance matrix

    Args:
        alpha (int/float): range in km
        theta (future dask array): great-circle distance in km
        lower (int): lower truncation
        n (int): natural number, n>0
        upper (upper truncation): upper truncation
        client 
    Returns:
        future dask array: Complex-Valued matrix
    """
    theta = client.scatter(theta)
    theta = client.persist(theta.result())

    lower_index = da.arange(lower, -1, 1)
    upper_index = da.arange(1, upper+1, 1)
    coefs_lower = da.zeros_like(theta)
    coefs_upper = da.zeros_like(theta)
    for k in lower_index:
        coefs_lower += da.float_power(abs(k), (-2*n)) * da.exp(k*theta*1j)
    for k in upper_index:
        coefs_upper += da.float_power(abs(k), (-2*n)) * da.exp(k*theta*1j)
    lower = client.scatter(coefs_lower)
    upper = client.scatter(coefs_upper)
    return client.submit(da.add, lower, upper)


def circular_matern(alpha, theta, nu, lower, upper, client):
    """Circular Matern covariance matrix

    Args:
        alpha (int/float): range in km
        theta (future dask array): great-circle distance matrix
        nu (float): smoothness
        lower (int): lower truncation
        upper (int): upper truncation
        client 

    Returns:
        future dask array: Complex-Valued Matrix 
    """
    theta = client.scatter(theta)
    theta = client.persist(theta.result())
    index = da.arange(lower, upper + 1, 1)
    coefs = da.zeros_like(theta)
    for k in index:
        coefs += (alpha**2 + k**2)**(-nu - .5)*da.exp(k*theta*1j)

    coefficients = client.scatter(coefs)
    return coefficients


# psi(theta)=sigma² * sum_{k=0} ^ N (alpha² + k²)^{-nu- .5} * P_k(cos(theta))


def legendre_matern(sigma, alpha, nu, theta, N, client):
    """_summary_

    Args:
        sigma (float): variance
        alpha (float/int): range, >0
        nu (float): smoothness, >0
        theta (array): great-circle distance
        N (int): truncation order
        client (_type_):

    Returns:
        covariance matrix of Legendre-Matern type: array
    """
    cos_theta = client.scatter(
        da.cos(theta))  # scatter to multiple workes cos(theta)
    cos_theta = client.gather(cos_theta)  # gather result
    cos_theta = cos_theta.persist()  # load into memory for calculations
    # coefficients [c_0,c_1,...,c_N] for sum c_0*P_0(cos(theta)) + ... + c_N*P_N(cos(theta))
    term = da.zeros_like(theta)
    coefs = np.zeros(N)
    for k in range(N):
        coefs[k] = (alpha**2 + k**2)**(-nu - .5)
        ###
        # c_0 = (alpha² + 0²)^(-nu - 0.5)
        # c_1 = (alpha² + 1²)^(-nu - 0.5)
        # .
        # .
        # .
        # c_N = (alpha²2 + N²)^(-nu - 0.5)
        ###
    legendre = client.submit(legval, cos_theta, coefs)
    ###
    # legendre = sum_{k=0} ^ N (alpha² + k²)^{-nu- .5} * P_k(cos(theta))
    #            = (alpha² + 0²)^(-nu - 0.5)*P_0(cos(theta)) + ... + (alpha²2 + N²)^(-nu - 0.5)*P_n(cos(theta))
    #            = c_0*P_0(cos(theta)) + ... + c_N*P_N(cos(theta))
    #            = sum_{k=0} ^ N c_k * P_k(cos(theta)) done by legvan(cos_theta,coefs)
    ###
    # just multiply legendre by sigma²
    return client.submit(da.multiply, sigma**2, legendre)


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
