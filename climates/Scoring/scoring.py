import numpy as np
import xarray as xr


def re_score(xbm_unstacked, xam_unstacked, xref_unstacked):
    """ We want to compute the Reduction of Error Score:

        RE_score = 1 - A/B

        where:
            -) A (mean squared error of analysis w.r.t xref) = 1/m * Σ_{i=1} ^ m (xam[i,:,:] - xref[i,:,:])² 
            -) B (mean squared error of background w.r.t. xref)= 1/m * Σ_{i=1} ^ m (xbm[i,:,:] - xref[i,:,:])²

    Args:
        xbm_unstacked (xarray.DataArray): initial ensemble anomaly mean, shape = (n_months,n_latitude,n_longitude)
        xam_unstacked (xarray.DataArray): updated ensemble anomaly mean, shape = (n_months,n_latitude,n_longitude)
        xref_unstacked (xarray.DataArray): reference dataset, shape = (n_months,n_latitude,n_longitude)

    Returns:
        RE_score (xarray.DataArray, numpy.array, dask.array): array of pointwise RE_scores
    """
    # select only points where we have a reference comparison

    non_updated = xbm_unstacked.where(np.logical_not(np.isnan(xref_unstacked)))
    updated = xam_unstacked.where(np.logical_not(np.isnan(xref_unstacked)))
    n_months = xbm_unstacked.shape[0]

    # get the A and B terms
    term_num = updated - xref_unstacked  # xam - xref
    term_den = non_updated - xref_unstacked  # xbm - xref
    # (xam - xref)² at points where values are not NaN's
    term_num_sq = np.power(
        term_num, 2, where=np.logical_not(np.isnan(term_num)))
    # (xbm - xref)² at points where values are not NaN's
    term_den_sq = np.power(
        term_den, 2, where=np.logical_not(np.isnan(term_den)))
    num = (1/n_months) * xr.DataArray.sum(term_num_sq, axis=0, skipna=False)
    den = (1/n_months) * xr.DataArray.sum(term_den_sq,
                                          axis=0, skipna=False)  # term 'B'
    re_score = 1 - \
        np.divide(num, den, where=np.logical_not(np.isnan(num)))  # 1 - A/B
    return re_score


def crps_ensembles_xarray(xa, xref):
    """ CRPS = 1/Ne * Σ_{i=1} ^ Ne |x^a[i,:] - xr[:]| - 1/(2*Ne²) Σ_{i,j=1} ^ Ne |x[i,:] - x[j,:]|

    Args:
        xa_unstacked (xr.DataArray): updated ensemble members, shape=(ensembles,months, latitude,longitude)
        xref_unstacked (xr.DataArray): reference dataset, shape=(months,latitude,longitude)

    Returns:
        crps (xr.DataArray): pointwise crps maps, shape (months,latitude,longitude)
    """
    updated = xa.where(np.logical_not(np.isnan(xref)))  # cast to common extend
    Ne = updated.shape[0]  # count number of ensemble members

    const_1 = 1/Ne  # 1/Ne
    const_2 = 1/(2*(Ne**2))  # 1/2*Ne²
    term = updated - xref  # xa-xref
    # here we can avoid looping by summing over the member_nr dimension
    first_sum = np.abs(xr.DataArray.sum(term, dim='member_nr', skipna=False))
    # define the list for the second sum loop (here we're forced to loop)
    second_sum = []
    for i in range(Ne):
        for j in range(Ne):
            x = np.abs(updated[i, :, :, :].drop_vars(
                'member_nr') - updated[j, :, :, :].drop_vars(
                'member_nr'))
            # note that here we drop the member_nr since we're taking differences between the ensemble members so the
            second_sum.append(x)
    second_sum = xr.DataArray.sum(xr.combine_nested(
        second_sum, concat_dim='new'), dim='new', skipna=False)
    crps = const_1 * first_sum - const_2 * second_sum
    return crps


def es_score_xarray(xa, xref):
    """ES = 1/Ne * Σ_{i=1} ^ Ne ||x^a[i,:] - xr[:]|| - 1/(2*Ne²) Σ_{i,j=1} ^ Ne ||x[i,:] - x[j,:]||

    Args:
        xa (xr.DataArray): stacked updated member, shape=(Ne,Nx) where Nx is the stacked dimension: stacked_dim=('latitude','longitude','time')
        xref (xr.DataArray): stacked reference dataset,stacked_dim=('latitude','longitude','time')
    """
    updated = xa.where(np.logical_not(np.isnan(xref)))  # cast to common extend
    Ne = updated.shape[0]  # count number of ensemble members

    const_1 = 1/Ne  # 1/Ne
    const_2 = 1/(2*(Ne**2))  # 1/2*Ne²
    term = updated - xref  # xa-xref
    # here we can avoid looping by summing over the member_nr dimension
    term = term.dropna(dim='stacked_dim')
    first_sum = np.sum(np.linalg.norm(term, axis=1))
    # define the list for the second sum loop (here we're forced to loop)
    second_sum = 0
    for i in range(Ne):
        for j in range(Ne):
            second_sum += np.linalg.norm(updated[i, :].dropna(dim='stacked_dim').drop_vars(
                'member_nr') - updated[j, :].dropna(dim='stacked_dim').drop_vars(
                'member_nr'))
            # note that here we drop the member_nr since we're taking differences between the ensemble members so the

    es = const_1 * first_sum - const_2 * second_sum
    return es
