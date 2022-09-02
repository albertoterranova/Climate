import xarray as xr
from climates.Utils.space_time_slicer import space_time_slicer as slicer
from climates.Utils.forward_operator import forward_operator
from climates.Kalman.ensrf import ensrf
import dask.array as da
from dask.distributed import Client, LocalCluster
from climates.Covariance.covariance_kernels_sphere import *
from climates.Covariance.IWP import IWP
from climates.Covariance.shrinkage import *
from climates.Distances.haversine_dask import *
import numpy as np
from climates.Scoring.scoring import *
from climates.Plot.plot import *

cluster = LocalCluster()
client = Client(cluster)
client.dashboard_link

base_folder = "/home/alberto/Desktop/climates/Data"

ens = xr.open_dataset(base_folder+"/ensemble.nc")
mean = xr.open_dataset(base_folder+"/ensemble_mean.nc")
obs = xr.open_dataset(base_folder+"/observations.nc")
ref = xr.open_dataset(base_folder+"/truth.nc")

xb, xbm, y, xref = slicer(ens, mean, obs, ref, [
                          '1961-01-16', '1961-01-16'], stacked=False)
xb1, xbm1, y1, xref1, _, _, _, _, _, _ = slicer(
    ens, mean, obs, ref, ['1961-01-16', '1961-01-16'], type='dask')
xb2, xbm2, y2, xref2 = slicer(
    ens, mean, obs, ref, ['1961-01-16', '1961-01-16'])

# subset every 6 element horizontally and vertically
xm = xbm[:, ::6, ::6].stack(stacked_dim=('latitude', 'longitude', 'time'))
x = xb[:, :, ::6, ::6].stack(stacked_dim=('latitude', 'longitude', 'time')).T
yo = y[:, ::6, ::6].stack(stacked_dim=('latitude', 'longitude', 'time'))
H = forward_operator(xbm[:, ::6, ::6], y[:, ::6, ::6], 1)
H = da.from_array(H)


xa, xap, xam = ensrf(xm, x, yo, H, chunk_size=55, obs_error=0.9, client=client)
xa = client.persist(xa)
xam = client.persist(xam)
xap = client.persist(xap)

lat, lon = xm.latitude, xm.longitude
latlon = np.vstack((lat, lon)).T
latlon = da.from_array(latlon)
gcd = haversine_vector_dask(latlon, latlon, comb=True, client=client)

localization = powered_exp(1500, gcd, 0.7, client)

xa_loc, xap_loc, xam_loc = ensrf(
    xm, x, yo, H, chunk_size=55, obs_error=0.9, localization=localization, client=client)

xbp = x - xm
P_lw = lw(da.from_array(xbp.values), client)
P_rblw = rblw(da.from_array(xbp.values), client)
P_ka = ka(da.from_array(xbp.values), localization, client)
P_iwp = IWP(da.from_array(xbp.values), localization, 1000, client)

xa_lw, xap_lw, xam_lw = ensrf(
    xm, x, yo, H, chunk_size=55, obs_error=0.9, P=P_lw, client=client)
xa_rblw, xap_rblw, xam_rblw = ensrf(
    xm, x, yo, H, chunk_size=55, obs_error=0.9, P=P_rblw, client=client)
xa_ka, xap_ka, xam_ka = ensrf(
    xm, x, yo, H, chunk_size=55, obs_error=0.9, P=P_ka, client=client)
xa_iwp, xap_iwp, xam_iwp = ensrf(
    xm, x, yo, H, chunk_size=55, obs_error=0.9, P=P_iwp, client=client)


ref = xref[:, ::6, ::6]

re = re_score(xm.unstack(), xam.unstack(), ref.T)
re.compute()

crps = crps_ensembles_xarray(xa.unstack(), ref.T)

es = es_score_xarray(xa.T, ref.stack(
    stacked_dim=('latitude', 'longitude', 'time')))

plot_base_map(crps)
plot_2_vs_2(xam,xam_loc)
plot_3_vs_3(xm,xam,ref)
