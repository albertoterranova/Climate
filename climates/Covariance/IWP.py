import dask.array as da


def IWP(samples, scale_matrix, dof, client):
    """Inverse Wishart Prior Covariance Matrix

    https://en.wikipedia.org/wiki/Inverse-Wishart_distribution

    Args:
        samples (dask.array): normally distributed samples X ~ N(0,S), where X has dimension Nx x Ne with 
                                Nx being the state dimension and Ne being the number of samples
        scale_matrix (dask.array): scale matrix same shape as sample covariance
        dof (int): degrees of freedom
        client : dask client to handle lazy computations

    Returns : 
        posterior_mean (dask.array) : posterior mean of the IWP distribution
    """

    ne = samples.shape[1]

    nx = samples.shape[0]

    try:
        dof > nx - 1
    except ValueError:
        print(
            "Degrees of Freedom must be strintly greater than sample covariance dimension")

    psi = client.persist(scale_matrix)
    psi = client.scatter(psi, broadcast=True).result()

    # samples = client.persist(samples) optional since they probably aready loaded into memory
    samples = client.scatter(samples, broadcast=True).result()

    cov = da.cov(samples, rowvar=True, bias=True)

    cov = client.persist(cov)
    cov = client.scatter(cov, broadcast=True).result()

    posterior_mean = 1/(ne+dof-nx-1) * (ne*cov+psi)
    return posterior_mean
