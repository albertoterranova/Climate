import dask.array as da


def lw(x, client):
    """Ledoit-Wofl shrinkage

    Args:
        x (dask.array): Nx x Ne array of samples containing the difference around the mean i.e. x' = x - \bar(x)
        client (dask.client)

    Returns:
        shrunk (dask.array): Nx x Nx shrunk covariance matrix
    """
    x = client.persist(x)
    x = client.scatter(x, broadcast=True).result()

    N = x.shape[1]
    n = x.shape[0]

    P = 1/N * da.dot(x, x.T)
    P = client.persist(P)
    P = client.scatter(P, broadcast=True).result()

    num = 0
    for i in range(N):
        Delta = x[:, i] - x[:, i].T
        Delta = client.persist(Delta)
        Delta = client.scatter(Delta, broadcast=True).result()
        term = P - Delta
        term = client.persist(term)
        term = client.scatter(term, broadcast=True).result()

        norm = da.linalg.norm(term, 'fro')
        norm = client.persist(norm)
        norm = norm**2
        norm = client.persist(norm)

        num += norm
    squared_cov = da.square(P)
    s_cov = client.persist(squared_cov)
    s_cov = client.scatter(s_cov, broadcast=True).result()
    trace_s_cov = da.trace(s_cov)
    trace_s_cov = client.persist(trace_s_cov)

    trace_cov = da.trace(P)
    trace_cov = client.persist(trace_cov)

    den = N**2 * (trace_s_cov - ((trace_cov**2)/n))

    ratio = num/den

    alpha = da.min(ratio, 1)

    T = (trace_cov/n) * da.identity(n)
    T = client.persist(T)
    T = client.scatter(T, broadcast=True).result()

    shrunk = alpha*T + (1-alpha)*P
    shrunk = client.persist(shrunk)
    shrunk = client.scatter(shrunk, broadcast=True).result()
    return shrunk


def rblw(x, client):
    """Rao-Blackwellized-Ledoit-Wolf shrinkage
    Args:
        x (dask.array): Nx x Ne array of difference around the mean
        client (dask.client): Client

    Returns:
        shrunk (dask.array): Nx x Nx shrunk covariance matrix
    """
    x = client.persist(x)
    x = client.scatter(x, broadcast=True).result()
    N = x.shape[1]
    n = x.shape[0]

    P = 1/N * da.dot(x, x.T)
    P = client.persist(P)
    P = client.scatter(P, broadcast=True).result()

    squared_cov = da.square(P)
    s_cov = client.persist(squared_cov)
    s_cov = client.scatter(s_cov, broadcast=True).result()
    trace_s_cov = da.trace(s_cov)
    trace_s_cov = client.persist(trace_s_cov)

    trace_cov = da.trace(P)
    trace_cov = client.persist(trace_cov)

    num = ((N-2)/n) * trace_s_cov + trace_cov**2
    num = client.persist(num)

    den = (N+2) * (trace_s_cov - (trace_cov**2)/n)
    den = client.persist(den)

    ratio = num/den
    ratio = client.persist(ratio)

    alpha = da.min(ratio, 1)

    T = (trace_cov/n) * da.identity(n)
    T = client.persist(T)
    T = client.scatter(T, broadcast=True).result()

    shrunk = alpha*T + (1-alpha)*P
    shrunk = client.persist(shrunk)
    shrunk = client.scatter(shrunk, broadcast=True).result()
    return shrunk


def ka(x, T, client):
    """Knowledge Aided shrinkage

    Args:
        x (dask.array): Nx x Ne array of difference around the mean
        T (dask.array): Nx x Nx target matrix for the shrinkage
        client (dask.client): Client


    Returns:
        shrunk (dask.array): Nx x Nx shrunk covariance matrix
    """
    N = x.shape[1]
    n = x.shape[0]

    term = 0
    for i in range(N):
        Delta = x[:, i] - x[:, i].T
        Delta = client.persist(Delta)
        Delta = client.scatter(Delta, broadcast=True).result()
        norm = da.linalg.norm(term, 'fro')
        norm = client.persist(norm)
        norm = norm**4
        norm = client.persist(norm)

        term += norm

    P = 1/N * da.dot(x, x.T)
    P = client.persist(P)
    P = client.scatter(P, broadcast=True).result()
    norm_num = da.linalg.norm(P, 'fro')
    norm_num = client.persist(norm_num)

    num = (1/(N**2)) * term - (1/N) * norm_num**2
    num = client.persist(num)

    T = client.persist(T)
    T = client.scatter(T, broadcast=True).result()

    diff = P - T
    diff = client.persist(T)
    diff = client.scatter(T, broadcast=True).result()

    den = da.linalg.norm(diff, 'fro')**2
    den = client.persist(den)
    ratio = num/den

    alpha = da.min(ratio, 1)
    alpha = client.persist(alpha)

    shrunk = alpha*T + (1-alpha)*P
    shrunk = client.persist(shrunk)
    shrunk = client.scatter(shrunk, broadcast=True).result()
    return shrunk
