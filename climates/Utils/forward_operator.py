from utils import build_base_forward
from scipy.linalg import block_diag
import numpy as np


def forward_operator(xbm_unstacked, y_unstacked, months):
    matrix_list = []
    for i in range(months):
        matrix_list.append(build_base_forward(
            xbm_unstacked.isel(time=i), y_unstacked.isel(time=i)))
    H = block_diag(*matrix_list)
    H = H[np.logical_not(np.isnan(y_unstacked.isel(time=slice(months)).stack(
        stacked_dim=('latitude', 'longitude', 'time')))), :]

    return H


def forward_operator_nan(xbm_unstacked, y_unstacked, months):
    matrix_list = []
    for i in range(months):
        matrix_list.append(build_base_forward(
            xbm_unstacked.isel(time=i), y_unstacked.isel(time=i)))
    H = block_diag(*matrix_list)

    return H


def inverse_forward_operator(y_unstacked, xbm_unstacked, months):
    matrix_list = []
    for i in range(months):
        matrix_list.append(build_base_forward(
            y_unstacked.isel(time=i), xbm_unstacked.isel(time=i)))
    H = block_diag(*matrix_list)
    H = H[:, np.logical_not(np.isnan(y_unstacked.isel(time=slice(months)).stack(
        stacked_dim=('latitude', 'longitude', 'time'))))]
    return H
