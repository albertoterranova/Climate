from dask.array import from_array as fa
import numpy as np
import xarray as xr


def space_time_slicer(xbm, xb, y, xref, time_window, space_bounds=None, type='xarray', stacked='True'):
    """_summary_

    Args:
        time_window (list): e.g. ['1961-01-16','1961-06-16'] 
        space_bounds (1d-array, optional): [l0,l1,L0,L1] where
                                -) l0 : lower latitudinal bound in [-88.57,88.57]
                                -) l1 : upper latitudinal bound in [-88.57,88.57]
                                -) L0 : lower longitudinal bound in [-180,180]
                                -) L1 : upper longitudinal bound in [-180,180]
        xbm (xarray): Ensemble mean from the load function
        xb (xarray): Ensemble members from the load function
        y (xarray): Observations/Instrumeental from the load function
        xref (xarray): Reference Dataset from the load function
        type (str, optional): type of returned data:
                                1) 'xarray' returns data in xarray format
                                2) 'dask' returns data in dask array format
                                3) 'numpy' returns data in numpy array format
        stacked (boolean): Default=True returns vectors stacked in latitude longitude and time. 
                            If stacked=False the function returns the objects in original shape

    Returns:
        'xarray': returns the sliced dataset in xarray format, total of 4
        'dask' : returns the sliced dataset and the coordinates in dask array format, total of 10 in order:
                1) ensemble mean 
                2) ensemble members
                3) observations
                4) reference
                5) ensemble latitude 
                6) ensemble longitude
                7) observations latitude
                8) observations longitude
                9) reference latitude
                10) reference longitude
        'numpy' : returns the sliced dataset in numpy and the coordinates in numpy array format, total of 10 (same as dask)
    """
    t0 = time_window[0]  # set lower temporal bound
    t1 = time_window[1]  # set upper temporal bound
    # temporal slice ensemble mean
    xbm_unstacked = xbm.anomaly.sel(time=slice(t0, t1))
    # temporal slice ensemble members
    xb_unstacked = xb.anomaly.sel(time=slice(t0, t1))
    # temporal slice observations
    y_unstacked = y.anomaly.sel(time=slice(t0, t1))
    xref_unstacked = xref.anomaly.sel(
        time=slice(t0, t1))  # temporal slice reference
    if space_bounds is not None:  # slice over spatial bounds if any
        l0 = space_bounds[0]  # set lower latitudinal bound
        l1 = space_bounds[1]  # set upper latitudinal bound
        L0 = space_bounds[2]  # set lower longitudinal bound
        L1 = space_bounds[3]  # set upper longitudinal bound
        xbm_unstacked = xbm_unstacked.sel(latitude=slice(
            l1, l0), longitude=slice(L0, L1))  # spatial slice ensemble mean
        # note that here the longitude bounds are reversed since the latitude are in [88.57,-88.57]
        xb_unstacked = xb_unstacked.sel(latitude=slice(
            l1, l0), longitude=slice(L0, L1))  # spatial slice ensemble members
        # same latitude inversion as above
        y_unstacked = y_unstacked.sel(latitude=slice(
            l0, l1), longitude=slice(L0, L1))  # spatial slice observation
        xref_unstacked = xref_unstacked.sel(latitude=slice(
            l1, l0), longitude=slice(L0, L1))  # spatial slice reference
    else:  # is no spatial bounds returns the original datasets
        xbm_unstacked = xbm_unstacked
        xb_unstacked = xb_unstacked
        y_unstacked = y_unstacked
        xref_unstacked = xref_unstacked

    if stacked == 'True':  # if we want to stack the dataset
        xbm_stacked = xbm_unstacked.stack(
            stacked_dim=('latitude', 'longitude', 'time'))
        xb_stacked = xb_unstacked.stack(
            stacked_dim=('latitude', 'longitude', 'time'))
        y_stacked = y_unstacked.stack(
            stacked_dim=('latitude', 'longitude', 'time'))
        xref_stacked = xref_unstacked.stack(
            stacked_dim=('latitude', 'longitude', 'time'))
        if type == 'xarray':  # return the stacked xarray datasets
            return xbm_stacked, xb_stacked, y_stacked, xref_stacked
        elif type == 'dask':
            # return the stacked dask arrays
            # note that since we lose the xarray format with coordinates we return also the latitude and longitude dask array of each dataset
            return (fa(xbm_stacked.values), fa(xb_stacked.values), fa(y_stacked.values), fa(xref_stacked.values),
                    fa(xbm_stacked.latitude.values), fa(
                        xbm_stacked.longitude.values),
                    fa(y_stacked.latitude.values), fa(
                        y_stacked.longitude.values),
                    fa(xref_stacked.latitude.values), fa(xref_stacked.longitude.values))

        elif type == 'numpy':
            # return the stacked numpy arrays
            # here also we return the arrays of latitude and longitude
            return (xbm_stacked.values, xb_stacked.values, y_stacked.values, xref_stacked.values,
                    xbm_stacked.latitude.values, xbm_stacked.longitude.values,
                    y_stacked.latitude.values, y_stacked.longitude.values,
                    xref_stacked.latitude.values, xref_stacked.longitude.values)

    else:  # return the unstacked arrays in either xarray dask or numpy
        if type == 'xarray':
            return xbm_unstacked, xb_unstacked, y_unstacked, xref_unstacked
        elif type == 'dask':
            return (fa(xbm_unstacked.values), fa(xb_unstacked.values), fa(y_unstacked.values), fa(xref_unstacked.values),
                    fa(xbm_unstacked.latitude.values), fa(
                        xbm_unstacked.longitude.values),
                    fa(y_unstacked.latitude.values), fa(
                        y_unstacked.longitude.values),
                    fa(xref_unstacked.latitude.values), fa(xref_unstacked.longitude.values))
        elif type == 'numpy':
            return (xbm_unstacked.values, xb_unstacked.values, y_unstacked.values, xref_unstacked.values,
                    xbm_unstacked.latitude.values, xbm_unstacked.longitude.values,
                    y_unstacked.latitude.values, y_unstacked.longitude.values,
                    xref_unstacked.latitude.values, xref_unstacked.longitude.values)
