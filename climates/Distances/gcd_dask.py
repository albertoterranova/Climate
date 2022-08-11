import dask.array as da


def gcd_dask(lat, lon, client):
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
    x = client.submit(da.arctan2, i, m)
    y = client.submit(da.multiply, x, r)  # get distance in km
    return y
