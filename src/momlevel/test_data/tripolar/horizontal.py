""" test_data - module for generating test data """

import datetime as dt

import cftime
import numpy as np
import xarray as xr

__all__ = [
    "generate_test_data",
    "generate_test_data_dz",
    "generate_test_data_uv",
]


def xy_fields(dset=None, point="h"):

    dset = xr.Dataset if dset is None else dset

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
        np.random.normal(100.0, 10.0, (5, 5)),
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
