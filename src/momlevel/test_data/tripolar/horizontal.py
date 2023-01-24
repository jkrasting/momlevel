""" horizontal.py - module for generating xy test data """

import numpy as np
import xarray as xr

__all__ = [
    "xy_fields",
]


def xy_fields(dset=None, seed=123):
    """Function to set up a simple x-y grid

    This function sets up a simple horizontal grid with dimensions and axes
    in the style of MOM6. Returns dimensions, geolat/geolon coordinates,
    and a matching areacello field.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset, optional
        Existing dataset to append grid. If not specified, an empty
        dataset is initialized. By default, None
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        ntimes x 5 x 5 x 5 point dataset for unit testing
    """

    dset = xr.Dataset() if dset is None else dset

    dset["xh"] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="xh",
        attrs={
            "long_name": "h point nominal longitude",
            "units": "degrees_east",
            "axis": "X",
            "cartesian_axis": "X",
        },
    )

    dset["yh"] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="yh",
        attrs={
            "long_name": "h point nominal latitude",
            "units": "degrees_north",
            "axis": "Y",
            "cartesian_axis": "Y",
        },
    )

    # geolon / geolat as cell centers
    lon = np.arange(0.0, 361.0, 72.0)
    lat = np.arange(-90.0, 91.0, 36.0)

    lon = [(lon[x] + lon[x + 1]) / 2.0 for x in range(0, len(lon) - 1)]
    lat = [(lat[x] + lat[x + 1]) / 2.0 for x in range(0, len(lat) - 1)]
    geolon, geolat = np.meshgrid(lon, lat)

    dset["geolon"] = xr.DataArray(
        geolon,
        dims=("yh", "xh"),
        attrs={
            "long_name": "Longitude of tracer (T) points",
            "units": "degrees_east",
            "cell_methods": "time: point",
        },
    )

    dset["geolat"] = xr.DataArray(
        geolat,
        dims=("yh", "xh"),
        attrs={
            "long_name": "Latitude of tracer (T) points",
            "units": "degrees_north",
            "cell_methods": "time: point",
        },
    )

    areacello = xr.DataArray(
        np.random.default_rng(seed).normal(100.0, 10.0, (5, 5)),
        dims=({"yh": dset.yh, "xh": dset.xh}),
    )
    areacello = areacello / areacello.sum()
    dset["areacello"] = areacello * 3.6111092e14
    dset["areacello"].attrs = {
        "long_name": "Ocean Grid-Cell Area",
        "units": "m2",
        "cell_methods": "area:sum yh:sum xh:sum time: point",
        "standard_name": "cell_area",
    }
    return dset
