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

""" Here I add also how I computed the forward operator for 6 months, a bit ugly but works (re-implementation of your code, considering all points including NaN's)
# Localization Strategy (Ugly Version) : create great circle distance matrix for 1 month, then stack it to 6 months
1) Forward operator 1 month: use two functions match_dataset and build_base_forward
def match_datasets(base_dataset, dataset_tomatch):
    
    # Define original lat-lon grid
    # Creates new columns converting coordinate degrees to radians.
    lon_rad = np.deg2rad(base_dataset.longitude.values.astype(np.float32))
    lat_rad = np.deg2rad(base_dataset.latitude.values.astype(np.float32))
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    
    # Define grid to be matched.
    lon_tomatch = np.deg2rad(dataset_tomatch.longitude.values.astype(np.float32))
    lat_tomatch = np.deg2rad(dataset_tomatch.latitude.values.astype(np.float32))
    lat_tomatch_grid, lon_tomatch_grid = np.meshgrid(lat_tomatch, lon_tomatch,
            indexing='ij')
    
    # Put everything in two stacked lists (format required by BallTree).
    coarse_grid_list = np.vstack([lat_tomatch_grid.ravel().T, lon_tomatch_grid.ravel().T]).T
    
    ball = BallTree(np.vstack([lat_grid.ravel().T, lon_grid.ravel().T]).T, metric='haversine')
    
    distances, index_array_1d = ball.query(coarse_grid_list, k=1)
    
    # Convert back to kilometers.
    distances_km = 6371 * distances

    # Sanity check.
    print("Maximal distance to matched point: {} km.".format(np.max(distances_km)))
    
    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    index_array_2d = np.hstack(np.unravel_index(index_array_1d, lon_grid.shape))

    return index_array_1d, index_array_2d


def build_base_forward(model_dataset, data_dataset):
    
    index_array_1d, index_array_2d = match_datasets(model_dataset, data_dataset)

    model_lon_dim = len(model_dataset.longitude)
    model_lat_dim = len(model_dataset.latitude)
    data_lon_dim = len(data_dataset.longitude)
    data_lat_dim = len(data_dataset.latitude)

    G = np.zeros((data_lon_dim * data_lat_dim, model_lon_dim * model_lat_dim),
            np.float32)

    # Set corresponding cells to 1 (ugly implementation, could be better).
    for data_ind, model_ind in enumerate(index_array_1d):
        G[data_ind, model_ind] = 1.0

    return G

2) Create array for 6 months
def H_6_months(xbm1,y1):
    H_00 = build_base_forward(xbm1.isel(time=0),y1.isel(time=0))
    H_01 = build_base_forward(xbm1.isel(time=0),y1.isel(time=1))
    H_02 = build_base_forward(xbm1.isel(time=0),y1.isel(time=2))
    H_03 = build_base_forward(xbm1.isel(time=0),y1.isel(time=3))
    H_04 = build_base_forward(xbm1.isel(time=0),y1.isel(time=4))
    H_05 = build_base_forward(xbm1.isel(time=0),y1.isel(time=5))
    H0 = np.hstack((H_00,H_01,H_02,H_03,H_04,H_05))
    
    H_10 = build_base_forward(xbm1.isel(time=1),y1.isel(time=0))
    H_11 = build_base_forward(xbm1.isel(time=1),y1.isel(time=1))
    H_12 = build_base_forward(xbm1.isel(time=1),y1.isel(time=2))
    H_13 = build_base_forward(xbm1.isel(time=1),y1.isel(time=3))
    H_14 = build_base_forward(xbm1.isel(time=1),y1.isel(time=4))
    H_15 = build_base_forward(xbm1.isel(time=1),y1.isel(time=5))
    H1 = np.hstack((H_10,H_11,H_12,H_13,H_14,H_15))
    
    H_20 = build_base_forward(xbm1.isel(time=2),y1.isel(time=0))
    H_21 = build_base_forward(xbm1.isel(time=2),y1.isel(time=1))
    H_22 = build_base_forward(xbm1.isel(time=2),y1.isel(time=2))
    H_23 = build_base_forward(xbm1.isel(time=2),y1.isel(time=3))
    H_24 = build_base_forward(xbm1.isel(time=2),y1.isel(time=4))
    H_25 = build_base_forward(xbm1.isel(time=2),y1.isel(time=5))
    H2 = np.hstack((H_20,H_21,H_22,H_23,H_24,H_25))
    
    H_30 = build_base_forward(xbm1.isel(time=3),y1.isel(time=0))
    H_31 = build_base_forward(xbm1.isel(time=3),y1.isel(time=1))
    H_32 = build_base_forward(xbm1.isel(time=3),y1.isel(time=2))
    H_33 = build_base_forward(xbm1.isel(time=3),y1.isel(time=3))
    H_34 = build_base_forward(xbm1.isel(time=3),y1.isel(time=4))
    H_35 = build_base_forward(xbm1.isel(time=3),y1.isel(time=5))
    H3 = np.hstack((H_30,H_31,H_32,H_33,H_34,H_35))
    
    H_40 = build_base_forward(xbm1.isel(time=4),y1.isel(time=0))
    H_41 = build_base_forward(xbm1.isel(time=4),y1.isel(time=1))
    H_42 = build_base_forward(xbm1.isel(time=4),y1.isel(time=2))
    H_43 = build_base_forward(xbm1.isel(time=4),y1.isel(time=3))
    H_44 = build_base_forward(xbm1.isel(time=4),y1.isel(time=4))
    H_45 = build_base_forward(xbm1.isel(time=4),y1.isel(time=5))
    H4 = np.hstack((H_40,H_41,H_42,H_43,H_44,H_45))
    
    H_50 = build_base_forward(xbm1.isel(time=5),y1.isel(time=0))
    H_51 = build_base_forward(xbm1.isel(time=5),y1.isel(time=1))
    H_52 = build_base_forward(xbm1.isel(time=5),y1.isel(time=2))
    H_53 = build_base_forward(xbm1.isel(time=5),y1.isel(time=3))
    H_54 = build_base_forward(xbm1.isel(time=5),y1.isel(time=4))
    H_55 = build_base_forward(xbm1.isel(time=5),y1.isel(time=5))
    H5 = np.hstack((H_50,H_51,H_52,H_53,H_54,H_55))
    
    H = np.vstack((H0,H1,H2,H3,H4,H5))
    return H
    
3) Apply localization matrix:
    def localization_matrix_6_months(localizer):
        d = localizer
        RHO = np.block([[d, d, d, d, d, d], [d, d, d, d, d, d], [d, d, d, d, d, d],
                        [d, d, d, d, d, d], [d, d, d, d, d, d], [d, d, d, d, d, d]])
    return RHO
    
4) Build localization matrix:
    latitude = dataset_members.latitude
    longitude = dataset_members.longitude
    cluster = ...
    client = ...
    theta = gcd_dask_future(latitude,longitude,client) # great circle distance for 1 month
    localizer = spherical_dask(alpha=...,theta=theta, client = client) # spherical localization for 1 month
    localizer = localizer.result()
    # xbm1,y1 = dataset_mean.anomaly.unstack(),dataset_instrumenta.anomaly.unstack() # needed for building the forward matrix in my old setup
    # H = H_6_months(xbm1,y1) # build forward matrix for 6 months
    localization_matrix_6_months = client.submit(localization_matrix_6_months,localizer)  # stack the 1 month localization to the whole update window of 6 month
    """

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
    # ???? = arctan2(sqrt((cos(??1) * sin(????))?? +  (cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????))??),cos(??1) * cos(??2) * cos(????))

    cos_phi2 = client.submit(da.cos, phi2)  # cos(??2)
    cos_phi1 = client.submit(da.cos, phi1)  # cos(??2)
    cos_d_l = client.submit(da.cos, delta_lambda)  # cos(????)
    sin_d_l = client.submit(da.sin, delta_lambda)  # sin(????)
    sin_phi1 = client.submit(da.sin, phi1)  # sin(??1)
    sin_phi2 = client.submit(da.cos, phi2)  # cos(??2)
    a = client.submit(da.multiply, cos_phi2, sin_d_l)  # cos(??1) * sin(????)
    b = client.submit(da.power, a, 2)  # (cos(??1) * sin(????))??

    c = client.submit(da.multiply, cos_phi1, sin_phi2)  # cos(??1) * sin(??1)
    d = client.submit(da.multiply, cos_phi2, sin_phi1)  # cos(??2) * sin(??1)
    e = client.submit(da.multiply, d, cos_d_l)  # cos(??2) * sin(??1) * cos(????)
    # cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????)
    f = client.submit(da.subtract, c, e)
    # (cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????))??
    g = client.submit(da.power, f, 2)
    # (cos(??1) * sin(????))?? +  (cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????))??
    h = client.submit(da.add, b, g)
    # sqrt((cos(??1) * sin(????))?? +  (cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????))??)
    i = client.submit(da.sqrt, h)

    k = client.submit(da.multiply, sin_phi1, sin_phi2)  # sin(??1) * sin(??2)
    l = client.submit(da.multiply, cos_phi1, cos_phi2)  # cos(??1) * cos(??2)
    m = client.submit(da.multiply, l, cos_d_l)  # cos(??1) * cos(??2) * cos(????)

    # arctan2(sqrt((cos(??1) * sin(????))?? +  (cos(??1) * sin(??1) - cos(??2) * sin(??1) * cos(????))??),cos(??1) * cos(??2) * cos(????))
    x = client.submit(da.arctan2, 1, m)
    y = client.submit(da.multiply, x, r)  # get distance in km
    return y

def rao_blackwell_ledoit_wolf(S,client):
    n = S.shape[0]
    p = len(S)
    assert S.shape == (p, p)

    alpha = (n-2)/(n*(n+2))
    beta = ((p+1)*n - 2) / (n*(n+2))
    S2 = client.submit(da.dot,S,S)
    trace_S2 = client.submit(da.trace,S2)
    trace_S = client.submit(da.trace,S)
    trace_S_squared = client.submit(da.float_power,S,2)
    term = client.submit(da.true_divide,trace_S2,trace_S_squared)
    term = client.gather(term)
    U = ((p * term) - 1)
    rho = da.min(alpha + da.true_divide(beta,U), 1)
    mu = client.submit(da.true_divide,S,p)
    F = client.submit(da.multiply,mu,da.eye(p))
    a = client.submit(da.multiply,1-rho,S)
    b = client.submit(da.multiply,rho,F)
    c = client.submit(da.add,a,b)
    return c

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
    """ ??(??)=???? * ??_{k=0}^N (???? + k??)^{-?? - .5} * P_k(cos(??))
    Args:
        sigma (float): variance
        alpha (float/int): range, ??>0
        nu (float): smoothness, ??>0
        theta (array): great-circle distance
        N (int): truncation order
        client 

    Returns:
        covariance matrix of Legendre-Matern type: array
    """
    cos_theta = client.scatter(da.cos(theta)) # scatter to multiple workes cos(??)
    cos_theta = client.gather(cos_theta) # gather result
    cos_theta = cos_theta.persist() # load into memory for calculations
    term = da.zeros_like(theta) # coefficients [c_0,c_1,...,c_N] needed for sum: c_0*P_0(cos(??)) + ... + c_N*P_N(cos(??))
    coefs = np.zeros(N)
    for k in range(N):
        coefs[k] = (alpha**2 + k**2)**(-nu - .5) 
        ###
        # c_0 = (???? + 0??)^(-?? - 0.5) 
        # c_1 = (???? + 1??)^(-?? - 0.5)
        # .
        # .
        # .
        # c_N = (????2 + N??)^(-?? - 0.5)   
        ###
    legendre = client.submit(legval, cos_theta, coefs) 
    ###
    # legendre = ??_{k=0} ^ N (???? + k??)^{-?? - .5} * P_k(cos(??))
    #            = (???? + 0??)^(-?? - 0.5)*P_0(cos(??)) + ... + (???? + N??)^(-?? - 0.5)*P_n(cos(??))
    #            = c_0*P_0(cos(??)) + ... + c_N*P_N(cos(??))
    #            = ??_{k=0} ^ N c_k * P_k(cos(??)) done by legvan(cos_theta,coefs)
    ###
    return client.submit(da.multiply, sigma**2, legendre) ## just multiply legendre by sigma??



def powered_exp(alpha, theta, tau, client):  # exp{-(????)^??}
    """Powered Exponential covariance matrix

    Args:
        alpha(int/float): range in km, ??>0
        theta(future): great circle distance matrix
        tau(float): in (0,1]
        client
    Returns:
        future dask array
    """
    a = client.submit(da.multiply, theta, alpha)  # ????
    b = client.submit(da.float_power, a, tau)  # (????)^??
    c = client.submit(da.multiply, b, -1)  # -(????)^??
    f = client.submit(da.exp, c)  # exp{-(????)^??}
    return f


def spherical_dask(theta, alpha, client):  # (1+????/2)(1-????)??_+
    """Spherical covariance matrix 

    Args:
        theta (future): great circle distance matrix
        alpha (int): range, ??>0
        client

    Returns:
        future dask array
    """
    # make the mask for places farer than alpha km
    theta = client.submit(da.ma.masked_greater, theta, alpha)
    a = client.submit(da.multiply, theta, da.array(alpha))  # ????
    b = client.submit(da.subtract, da.array(1), a)  # (1-????)
    c = client.submit(da.power, b, 2)  # (1-????)??
    g = client.submit(da.maximum, da.array(0), c)  # (1-????)??_+
    d = client.submit(da.true_divide, a, 2)  # ????/2
    e = client.submit(da.add, da.array(1), d)  # (1+????/2)
    f = client.submit(da.multiply, g, e)  # (1+????/2)(1-????)??_+
    # set covariance to location far apart to 0
    f = client.submit(da.ma.filled, f, 0)
    return f


def cauchy_dask(alpha, theta, nu, tau, client):  # (1 + (????)^??)^???(??/??)
    """Cauchy Covariance matrix 

    Args:
        alpha (int):  ??>0
        theta (future): great circel distance matrix 
        nu (float): in (0,1]
        tau (float): ??>0
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.multiply, da.array(alpha), theta)  # ????
    b = client.submit(da.float_power, a, nu)  # (????)^??
    c = client.submit(da.add, da.array(1), b)  # 1 + (????)^??
    const = -tau/nu  # ???(??/??)
    d = client.submit(da.power, c, const)  # (1 + (????)^??)^???(??/??)
    return d


def multiquadric_dask(alpha, theta, tau, client):  # (1 ??? ??)^2?? /(1 + ???? ??? 2?? cos(??))^??
    """Multiquadric covariance matrix 

    Args:
        alpha (int): ??>0
        theta (future): great circle distance matrix 
        tau (float): ?? in (0,1)
        client 
    Returns:
        future dask array
    """
    a = (1 - tau)**(2*alpha)  # (1 ??? ??)^2??
    b = 1 + tau**2  # 1 + ????
    c = 2*tau  # 2??
    d = client.submit(da.cos, theta)  # cos(??)
    e = client.submit(da.multiply, c, d)  # 2?? cos(??)
    f = client.submit(da.add, b, e)  # 1 + ???? ??? 2?? cos(??)
    g = client.submit(da.power, f, alpha)  # (1 + ???? ??? 2?? cos(??))^??
    # (1 ??? ??)^2?? /(1 + ???? ??? 2?? cos(??))^??
    h = client.submit(da.true_divide, a, g)
    return h


def sine_dask(theta, nu, client):  # 1-(sin(??/2))^??
    """sine power covariance matrix 

    Args:
        theta (future): great circle distance matrix 
        nu (float): ?? in (0,2)
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.true_divide, theta, 2)  # ??/2
    b = client.submit(da.sin, a)  # sin(??/2)
    c = client.submit(da.power, b, nu)  # (sin(??/2))^??
    d = client.submit(da.subtract, da.array(1), c)  # 1-(sin(??/2))^??
    return d


def askey_dask(theta, alpha, tau, client):  # ((1 ??? ????)^??)_+
    """Askey Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (int): ??>0
        tau (float): ??>2
        client 

    Returns:
        future dask array
    """
    a = client.submit(da.multiply, da.array(alpha), theta)  # ????
    b = client.submit(da.subtract, 1, a)  # 1 ??? ????
    c = client.submit(da.power, b, tau)  # (1 ??? ????)^??
    d = client.submit(da.maximum, da.array(0), c)  # ((1 ??? ????)^??)_+
    return d


def c2_dask(theta, alpha, tau, client):  # (1 + ??????)((1 ??? ????)^??))_+
    """C2-Wendland Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (int): ??>1/??
        tau (float): ??>4
        client 

    Returns:
        future dask array
    """
    a = tau*alpha  # ????
    b = client.submit(da.multiply, a, theta)  # ??????
    c = client.submit(da.add, da.array(1), b)  # 1 + ??????

    d = client.submit(da.multiply, da.array(alpha), theta)  # ????
    e = client.submit(da.subtract, da.array(1), d)  # 1 ??? ????
    f = client.submit(da.float_power, e, tau)  # (1 ??? ????)^??)
    g = client.submit(da.maximum, 0, f)  # ((1 ??? ????)^??))_+
    h = client.submit(da.multiply, c, g)  # (1 + ??????)((1 ??? ????)^??))_+
    return h


def c4_dask(theta, alpha, tau, client):  # (1 + ?????? + (???? - 1)/3 * (????)?? )((1 ??? ????)^??))_+
    """C4-Wendland Covariance Matrix

    Args:
        theta (future): great circle distance matrix
        alpha (float): ??>1/??
        tau (float): ??>6
        client 

    Returns:
        future dask array
    """
    a = tau*alpha  # ????
    b = client.submit(da.multiply, a, theta)  # ??????
    c = client.submit(da.add, da.array(1), b)  # 1 + ??????
    i = (tau**2 - 1)/3  # (???? - 1)/3
    d = client.submit(da.multiply, da.array(alpha), theta)  # ????
    l = client.submit(da.power, d, 2)  # (????)??
    m = client.submit(da.multiply, i, l)  # (???? - 1)/3 * (????)??
    n = client.submit(da.add, c, m)  # 1 + ?????? + (???? - 1)/3 * (????)??

    e = client.submit(da.subtract, da.array(1), d)  # 1 ??? ????
    f = client.submit(da.float_power, e, tau)  # ((1 ??? ????)^??))_+
    g = client.submit(da.maximum, 0, f)  # ((1 ??? ????)^??))_+
    # (1 + ?????? + (???? - 1)/3 * (????)?? )((1 ??? ????)^??))_+
    h = client.submit(da.multiply, c, g)

    return h


def poisson_dask(theta, alpha, client):  # (1-????)/((1-2??*cos(??) + ????)^(3/2))
    """Poisson Covariance Matrix

    Args:
        theta (future): great circle distance matrix 
        alpha (float): ?? in (0,1)
        client 

    Returns:
        future dask array
    """

    a = 1 - alpha**2  # (1-????)
    b = client.submit(da.cos, theta)  # cos(??)
    c = client.submit(da.multiply, da.array(2*alpha), b)  # 2??*cos(??)
    d = client.submit(da.subtract, da.array(1+alpha**2), c)  # 1-2??*cos(??) + ????
    e = client.submit(da.float_power, d, 1.5)  # (1-2??*cos(??) + ????)^(3/2)

    # (1-????)/((1-2??*cos(??) + ????)^(3/2))
    f = client.submit(da.true_divide, a, e)

    return f


# 1 - (??(xam - xref)??)/(??(xbm - xref)??) where the sum runs from 1 to 6 (months)
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
    b = client.submit(da.power, a, 2)  # (xam - xref)??
    c = client.submit(da.sum, b, axis=0)  # ??(xam - xref)??

    d = client.submit(da.subtract, xbm, xr)  # xbm - xref
    e = client.submit(da.power, d, 2)  # (xbm - xref)??
    f = client.submit(da.sum, e, axis=0)  # ??(xbm - xref)??

    g = client.submit(da.true_divide, c, f)  # ??(xam - xref)??/(??(xbm - xref)??

    # 1 - (??(xam - xref)??)/(??(xbm - xref)??)
    h = client.submit(da.subtract, da.array(1), g)

    # takes the space-time mean excluding the NaN's
    i = client.submit(da.nanmean, h)
    return h, i
