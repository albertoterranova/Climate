# Climate
Climate of the past, Data Assimilation with Ensemble Square Root (Offline) Filter in Dask

## Covariance
The folder /Covariance contains:
- IWP.py - Inverse Wishart Prior
- covariance_kernels_sphere.py - Positive Definite Functions on The Sphere
- shrinkage.py - Ledoit Wolf, Rao-Blackwellized-Ledoit-Wolf, Knowledge Aided


## Cython

The folder /Cython contains:
- .pyx Cython compiled positive definite functions on the sphere used for localizations
- .py file to install the Cython functions
Make sure you have a gcc compiler installed. 
**TODO**:
```sh
cd YOUR_FOLDER/climates/Cython
python file_name.py install
```
The main file is localizations.pyx which contains all the functions, thus the main compiler will be localizations.py
All the other files are just the single functions.

## Data

The folder /Data contains sample .nc files for testing purposes

- ensemble.nc is a NetCDF4 file containing 30 simulated samples of temperature fields 
- ensemble_mean.nc is their sample ensemble_mean
- observations.nc contains the proxy observations
- truth.nc contains the .ground truth

## Distances

The folder /Distances contains two different functions for calculating the great circle distance matrix

- gcd_dask.py uses the Vincenty formula
- haversine_dask.py uses the Sklearn haversine distance implemented in Dask

## Kalman

The folder /Kalman contains:

- ensrf.py - Ensemble Kalman Fitting

## Plot

The folder /Plot contains:

- plot.py - Basemap Wrapper for plotting utilities

## Scoring

The folder /scoring contains:

- scoring.py 
    - Reduction of Error (RE) score
    - Continuous Rank Probability Score (CRPS)
    - Energy Score (ES)

## Utils

The folder /Utils contains:

- LOAD.py - data loader for .nc files
- forward_operator.py - Create the forward (observation) matrix that maps the state space to the observational observational one
- space_time_slicer.py - slice the xarray in space and time, returns values in xarray, numpy and dask.
