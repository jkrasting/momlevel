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
    z_i = xr.DataArray(np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]), dims=("z_i"))
    z_l = xr.DataArray(np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]), dims=("z_l"))
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

    deptho = np.array(
        [
            np.random.uniform(0.0, 5.0, 5),
            np.random.uniform(0.0, 15.0, 5),
            np.random.uniform(0.0, 185.0, 5),
            np.random.uniform(0.0, 1815.0, 5),
            np.random.uniform(0.0, 6185.0, 5),
        ]
    )

    deptho = xr.DataArray(deptho, dims=({"yh": yh, "xh": xh}))
    areacello = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)), dims=({"yh": yh, "xh": xh})
    )
    areacello = areacello / areacello.sum()
    areacello = areacello * 3.6111092e14

    dset = xr.Dataset(
        {
            "thetao": thetao,
            "so": so,
            "volcello": volcello,
            "areacello": areacello,
            "deptho": deptho,
        }
    )
    dset = dset.assign_coords(
        {"time": time, "z_l": z_l, "z_i": z_i, "yh": yh, "xh": xh}
    )

    return dset


def generate_test_data_dz():
    """Function to generate test dataset for partial bottom cells

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    xh = np.arange(1, 6)
    xh = xr.DataArray(xh, dims=("xh"))

    yh = np.arange(10, 60, 10)
    yh = xr.DataArray(yh, dims=("yh"))

    np.random.seed(123)

    deptho = np.random.uniform(0.0, 100.0, (5, 5))
    deptho[2, 2] = np.nan
    deptho[2, 3] = np.nan
    deptho = xr.DataArray(deptho, coords=(yh, xh))

    z_i = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    z_i = xr.DataArray(z_i, dims=("z_i"))

    z_l = np.array((z_i[1::] + z_i[0:-1]) / 2.0)
    z_l = xr.DataArray(z_l, dims=("z_l"))

    dset = xr.Dataset({"deptho": deptho, "z_l": z_l, "z_i": z_i})

    return dset
