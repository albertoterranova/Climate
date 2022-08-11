import xarray as xr
import os
import numpy as np
from sklearn.neighbors import BallTree


def load(base_folder, TOT_ENSEMBLES_NUMBER, client):
    ens_mean_folder = os.path.join(base_folder, "Ensembles/Means/")
    ens_mem_folder = os.path.join(base_folder, "Ensembles/Members/")
    instrumental_path = os.path.join(
        base_folder, "Instrumental/HadCRUT.4.6.0.0.median.nc")
    reference_path = os.path.join(
        base_folder, "Reference/cru_ts4.05.1901.2020.tmp.dat.nc")

    dataset_instrumental = xr.open_dataset(instrumental_path)
    dataset_reference = xr.open_dataset(reference_path)

    # Rename so dimensions names agree with the other datasets.
    dataset_reference = dataset_reference.rename(
        {'lat': 'latitude', 'lon': 'longitude'})

    dataset_mean = xr.open_mfdataset(ens_mean_folder + '*.nc', concat_dim="time", combine="nested",
                                     data_vars='minimal', coords='minimal',
                                     compat='override')

    # Loop over members folders and merge.

    datasets = []
    for i in range(1, int(TOT_ENSEMBLES_NUMBER) + 1):
        current_folder = os.path.join(ens_mem_folder, "member_{}/".format(i))
        print(current_folder)
        datasets.append(xr.open_mfdataset(current_folder + '*.nc', concat_dim="time", combine="nested",
                                          data_vars='minimal', coords='minimal',
                                          compat='override'))

    dataset_members = xr.combine_by_coords(datasets)
    # Otherwise just return a dummy copy.

    # --------------
    # POSTPROCESSING
    # --------------

    # Drop the unused variables for clarity.
    dataset_mean = dataset_mean.drop(['cycfreq', 'blocks',
                                      'lagrangian_tendency_of_air_pressure',
                                      'northward_wind',
                                      'eastward_wind', 'geopotential_height',
                                      'air_pressure_at_sea_level',
                                      'total precipitation',
                                      'pressure_level_wind', 'pressure_level_gph'])
    dataset_members = dataset_members.drop(['cycfreq', 'blocks',
                                            'lagrangian_tendency_of_air_pressure',
                                            'northward_wind',
                                            'eastward_wind', 'geopotential_height',
                                            'air_pressure_at_sea_level',
                                            'total precipitation',
                                            'pressure_level_wind', 'pressure_level_gph'])

    dataset_reference = dataset_reference.drop(['stn'])
    dataset_instrumental = dataset_instrumental.drop(['field_status'])

    # Rename.
    dataset_mean = dataset_mean.rename(
        {'air_temperature': 'temperature'})
    dataset_members = dataset_members.rename(
        {'air_temperature': 'temperature'})
    dataset_reference = dataset_reference.rename(
        {'tmp': 'temperature'})
    dataset_instrumental = dataset_instrumental.rename(
        {'temperature_anomaly': 'anomaly'})

    # Convert longitudes to the standard [-180, 180] format.
    dataset_mean = dataset_mean.assign_coords(
        longitude=(((dataset_mean.longitude + 180) % 360) - 180)).sortby('longitude')
    dataset_members = dataset_members.assign_coords(
        longitude=(((dataset_members.longitude + 180) % 360) - 180)).sortby('longitude')

    # Match time index: except for the instrumental dataset, all are gridded
    # monthly, with index that falls on the middle of the month. Problem is
    # that February (middle is 15 or 16) is treated differently across the
    # datasets.

    # First convert everything to datetime.
    # Subset to valid times. This is because np.datetime only allows dates back to 1687,
    # but on the other hand cftime does not support the full set of pandas slicing features.

    dataset_mean['time'] = dataset_mean.indexes['time'].to_datetimeindex()
    dataset_members['time'] = dataset_members.indexes['time'].to_datetimeindex()

    # Use the index from the mean dataset, matching to the nearest timestamp.
    dataset_reference = dataset_reference.reindex(
        time=dataset_mean.time.values, method='nearest')
    dataset_instrumental = dataset_instrumental.reindex(
        time=dataset_mean.time.values, method='nearest')

    # Regrid the reference to the simulation grid (lower resolution).
    dataset_reference = dataset_reference.interp(latitude=dataset_mean["latitude"],
                                                 longitude=dataset_mean["longitude"])

    # Cast to common dtype.
    dataset_reference['temperature'] = dataset_reference['temperature'].astype(
        np.float32)
    dataset_reference = dataset_reference.sortby('time')
    monthly_avg_mean = dataset_mean.temperature.groupby(
        'time.month').mean(dim='time')
    monthly_avg_members = dataset_members.temperature.groupby(
        'time.month').mean(dim='time')
    monthly_avg_reference = dataset_reference.temperature.sel(time=slice(
        '1961-01-16', '1990-12-16')).groupby('time.month').mean(dim='time')
    # Compute difference wrt mean.
    dataset_mean['anomaly'] = dataset_mean.temperature.groupby(
        'time.month') - monthly_avg_mean
    dataset_members['anomaly'] = dataset_members.temperature.groupby(
        'time.month') - monthly_avg_members
    anomaly = dataset_reference.groupby('time.month') - monthly_avg_reference
    dataset_reference['anomaly'] = anomaly.temperature
    dataset_mean['baseline_mean'] = monthly_avg_mean
    dataset_members['baseline_mean'] = monthly_avg_members
    dataset_reference['baseline_mean'] = monthly_avg_reference

    dataset_instrumental['anomaly'] = dataset_instrumental['anomaly'].chunk()
    dataset_members['anomaly'] = dataset_members['anomaly'].chunk({
                                                                  'time': 480})
    dataset_mean['time'] = np.sort(dataset_mean['time'].values)
    dataset_members['time'] = np.sort(dataset_members['time'].values)
    dataset_instrumental['time'] = np.sort(dataset_instrumental['time'].values)
    dataset_reference['time'] = np.sort(dataset_reference['time'].values)
    dataset_instrumental = dataset_instrumental.chunk()

    dataset_reference = dataset_reference.chunk()

    try:
        dataset_mean = client.persist(dataset_mean)
    except:
        print('Error dataset mean')
    try:
        dataset_members = client.persist(dataset_members)
    except:
        print('Error dataset memebrs')
    try:
        dataset_instrumental = client.persist(dataset_instrumental)
    except:
        print('Error dataset instrumental')
    try:
        dataset_reference = client.persist(dataset_reference)
    except:
        print('Error dataset reference')

    return dataset_mean, dataset_members, dataset_instrumental, dataset_reference


def match_datasets(base_dataset, dataset_tomatch):
    """" Match two datasets defined on different grid.
    Given a base dataset and a dataset to be matched, find for each point in
    the dataset to mathc the closest cell in the base dataset and return its
    index.
    Parameters
    ----------
    base_dataset: xarray.Dataset
    dataset_tomatch: xarray.Dataset
    """
    # Define original lat-lon grid
    # Creates new columns converting coordinate degrees to radians.
    lon_rad = np.deg2rad(base_dataset.longitude.values.astype(np.float32))
    lat_rad = np.deg2rad(base_dataset.latitude.values.astype(np.float32))
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing='ij')

    # Define grid to be matched.
    lon_tomatch = np.deg2rad(
        dataset_tomatch.longitude.values.astype(np.float32))
    lat_tomatch = np.deg2rad(
        dataset_tomatch.latitude.values.astype(np.float32))
    lat_tomatch_grid, lon_tomatch_grid = np.meshgrid(lat_tomatch, lon_tomatch,
                                                     indexing='ij')

    # Put everything in two stacked lists (format required by BallTree).
    coarse_grid_list = np.vstack(
        [lat_tomatch_grid.ravel().T, lon_tomatch_grid.ravel().T]).T

    ball = BallTree(
        np.vstack([lat_grid.ravel().T, lon_grid.ravel().T]).T, metric='haversine')

    distances, index_array_1d = ball.query(coarse_grid_list, k=1)

    # Convert back to kilometers.
    distances_km = 6371 * distances

    # Sanity check.
    print("Maximal distance to matched point: {} km.".format(np.max(distances_km)))

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    index_array_2d = np.hstack(
        np.unravel_index(index_array_1d, lon_grid.shape))

    return index_array_1d, index_array_2d


def build_base_forward(model_dataset, data_dataset):
    """ Build the forward operator mapping a model dataset to a data dataset.
    This builds the generic forward matrix that maps points in the model space
    to points in the data space, considering both as static, uniform grids. 
    When used in practice one should remove lines of the matrix that correspond
    to NaN values in the data space.
    """
    index_array_1d, index_array_2d = match_datasets(
        model_dataset, data_dataset)

    model_lon_dim = model_dataset.dims['longitude']
    model_lat_dim = model_dataset.dims['latitude']
    data_lon_dim = data_dataset.dims['longitude']
    data_lat_dim = data_dataset.dims['latitude']

    G = np.zeros((data_lon_dim * data_lat_dim, model_lon_dim * model_lat_dim),
                 np.float32)

    # Set corresponding cells to 1 (ugly implementation, could be better).
    for data_ind, model_ind in enumerate(index_array_1d):
        G[data_ind, model_ind] = 1.0

    return G
