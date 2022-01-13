""" test_data.py - module for generating test data """

import xarray as xr
import numpy as np

__all__ = ["generate_test_data"]


def generate_test_data():
    """Function to generate dataset for unit testing

    Returns
    -------
    xarray.core.dataset.Dataset
        5x5x5x5 point dataset for unit testing
    """
    np.random.seed(123)
    time = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"))
    z_l = xr.DataArray(np.array([2.5, 50.0, 100.0, 1000.0, 5000.0]), dims=("z_l"))
    xh = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="xh")
    yh = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="yh")

    thetao = xr.DataArray(
        np.random.normal(15.0, 5.0, (5, 5, 5, 5)),
        dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
    )
    so = xr.DataArray(
        np.random.normal(35.0, 1.5, (5, 5, 5, 5)),
        dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
    )
    volcello = xr.DataArray(
        np.random.normal(1000.0, 100.0, (5, 5, 5, 5)),
        dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
    )
    areacello = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)), dims=({"yh": yh, "xh": xh})
    )
    areacello = areacello / areacello.sum()
    areacello = areacello * 3.6111092e14

    dset = xr.Dataset(
        {"thetao": thetao, "so": so, "volcello": volcello, "areacello": areacello}
    )
    dset = dset.assign_coords({"time": time, "z_l": z_l, "yh": yh, "xh": xh})

    return dset
