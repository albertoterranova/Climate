import dask.array as da


# mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
EARTH_RADIUS_KM = 6371.0088


def haversine_vector_dask(array1, array2,  client, comb=True):
    '''
    The exact same function as "haversine", except that this
    version replaces math functions with numpy functions.
    This may make it slightly slower for computing the haversine
    distance between two points, but is much faster for computing
    the distance between two vectors of points due to vectorization.
    '''

    # ensure will be able to iterate over rows by adding dimension if needed
    if array1.ndim == 1:
        array1 = da.expand_dims(array1, 0)
    if array2.ndim == 1:
        array2 = da.expand_dims(array2, 0)

    # Asserts that both arrays have same dimensions if not in combination mode
    if not comb:
        if array1.shape != array2.shape:
            raise IndexError(
                "When not in combination mode, arrays must be of same size. If mode is required, use comb=True as argument.")

    # unpack latitude/longitude
    lat1, lng1 = array1[:, 0], array1[:, 1]
    lat2, lng2 = array2[:, 0], array2[:, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = da.radians(lat1)
    lat1 = client.persist(lat1)
    lat1 = client.scatter(lat1, broadcatst=True).result()

    lng1 = da.radians(lng1)
    lng1 = client.persist(lng1)
    lng1 = client.scatter(lng1, broadcatst=True).result()

    lat2 = da.radians(lat2)
    lat2 = client.persist(lat2)
    lat2 = client.scatter(lat2, broadcatst=True).result()

    lng2 = da.radians(lng2)
    lng2 = client.persist(lng2)
    lng2 = client.scatter(lng2, broadcatst=True).result()

    # If in combination mode, turn coordinates of array1 into column vectors for broadcasting
    if comb:
        lat1 = da.expand_dims(lat1, axis=0)
        lat1 = client.persist(lat1)
        lat1 = client.scatter(lat1, broadcatst=True).result()
        lng1 = da.expand_dims(lng1, axis=0)
        lng1 = client.persist(lng1)
        lng1 = client.scatter(lng1, broadcatst=True).result()
        lat2 = da.expand_dims(lat2, axis=1)
        lat2 = client.persist(lat2)
        lat2 = client.scatter(lat2, broadcatst=True).result()

        lng2 = da.expand_dims(lng2, axis=1)
        lng2 = client.persist(lng2)
        lng2 = client.scatter(lng2, broadcatst=True).result()

    # calculate haversine
    lat = lat2 - lat1
    lat = client.persist(lat)
    lat = client.scatter(lat, broadcast=True).result()
    lng = lng2 - lng1
    lng = client.persist(lng)
    lng = client.scatter(lng, broadcast=True).result()

    "We want to compute the following quantity:\
        d = sin(lat/2)² + cos(lat1)*cos(lat2)*sin(lng/2)²"

    "As usual we decompose the sum into intermediate steps and persist the calculation into memory"
    "Note that we need to scatter each intermediate step thus to have a less burdensome calculation"
    a = da.sin(lat * 0.5) ** 2
    a = client.persist(a)
    a = client.scatter(a, broadcast=True).result()

    b = da.cos(lat1)
    b = client.persist(b)
    b = client.scatter(b, broadcast=True).result()

    c = da.cos(lat2)
    c = client.persist(c)
    c = client.scatter(c, broadcast=True).result()

    d = da.sin(lng * 0.5) ** 2
    d = client.persist(d)
    d = client.scatter(d, broadcast=True).result()

    e = b*c*d
    e = client.persist(e)
    e = client.scatter(e, broadcast=True).result()

    f = a + e
    f = client.persist(f)
    f = client.scatter(f, broadcast=True).result()

    g = da.arcsin(da.sqrt(f))
    g = client.persist(g)

    return 2 * EARTH_RADIUS_KM * g
