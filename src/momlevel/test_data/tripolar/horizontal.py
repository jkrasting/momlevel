""" horizontal.py - module for generating xy test data """

import numpy as np
import xarray as xr

__all__ = [
    "xy_fields",
]


def xy_fields(dset=None, point="h", seed=123):
    """Function to set up a simple x-y grid

    This function sets up a simple horizontal grid with dimensions and axes
    in the style of MOM6. Returns dimensions, geolat/geolon coordinates,
    and a matching areacello field.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset, optional
        Existing dataset to append grid. If not specified, an empty
        dataset is initialized. By default, None
    point : str, optional
        Staggered grid point of either "h", "u", "v", or "c".
        By default, "h"
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        ntimes x 5 x 5 x 5 point dataset for unit testing
    """

    dset = xr.Dataset() if dset is None else dset

    point_dict = {
        "h": ("xh", "yh", "geolon", "geolat", "areacello", "tracer (T)"),
        "u": (
            "xq",
            "yh",
            "geolon_u",
            "geolat_u",
            "areacello_cu",
            "zonal velocity (Cu)",
        ),
        "v": (
            "xh",
            "yq",
            "geolon_v",
            "geolat_v",
            "areacello_cv",
            "meridional velocity (Cv)",
        ),
        "c": ("xq", "yq", "geolon_c", "geolat_c", "areacello_bu", "corner (Bu)"),
    }

    attrs = point_dict[point]

    dset[attrs[0]] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=attrs[0],
        attrs={
            "long_name": f"{attrs[0][-1]} point nominal longitude",
            "units": "degrees_east",
            "axis": "X",
            "cartesian_axis": "X",
        },
    )

    dset[attrs[1]] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=attrs[1],
        attrs={
            "long_name": f"{attrs[1][-1]} point nominal latitude",
            "units": "degrees_north",
            "axis": "Y",
            "cartesian_axis": "Y",
        },
    )

    # geolon / geolat
    lon = np.arange(0.0, 361.0, 72.0)
    lat = np.arange(-90.0, 91.0, 36.0)

    lon = [(lon[x] + lon[x + 1]) / 2.0 for x in range(0, len(lon) - 1)]
    lat = [(lat[x] + lat[x + 1]) / 2.0 for x in range(0, len(lat) - 1)]
    geolon, geolat = np.meshgrid(lon, lat)

    dset[attrs[2]] = xr.DataArray(
        geolon,
        dims=(attrs[1], attrs[0]),
        attrs={
            "long_name": f"Longitude of {attrs[5]} points",
            "units": "degrees_east",
            "cell_methods": "time: point",
        },
    )

    dset[attrs[3]] = xr.DataArray(
        geolat,
        dims=(attrs[1], attrs[0]),
        attrs={
            "long_name": f"Latitude of {attrs[5]} points",
            "units": "degrees_north",
            "cell_methods": "time: point",
        },
    )

    areacello = xr.DataArray(
        np.random.default_rng(seed).normal(100.0, 10.0, (5, 5)),
        dims=({attrs[1]: dset[attrs[1]], attrs[0]: dset[attrs[0]]}),
    )
    areacello = areacello / areacello.sum()
    dset[attrs[4]] = areacello * 3.6111092e14
    dset[attrs[4]].attrs = {
        "long_name": "Ocean Grid-Cell Area",
        "units": "m2",
        "cell_methods": f"area:sum {attrs[1]}:sum {attrs[0]}:sum time: point",
        "standard_name": "cell_area",
    }
    return dset
