""" vertical.py - module for generating z-level test data """

import numpy as np
import xarray as xr

from .horizontal import xy_fields

__all__ = [
    "zlevel_fields",
]


def zlevel_fields(dset=None, include_deptho=True, seed=123):
    """Function to set up a simple z-level vertical grid

    This function sets up a simple vertical grid with dimensions and axes
    in the style of MOM6. Returns depth level centers and interfaces
    and an optional matching deptho field.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset, optional
        Existing dataset to append grid. If not specified, an empty
        dataset is initialized. By default, None
    include_deptho : bool, optional
        Include a matching deptho field. By default, True
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        5-vertical-level dataset for unit testing
    """
    dset = xr.Dataset() if dset is None else dset

    dset["z_i"] = xr.DataArray(
        np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]),
        dims=("z_i"),
        attrs={
            "long_name": "Depth at interface",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
        },
    )

    dset["z_l"] = xr.DataArray(
        np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]),
        dims=("z_l"),
        attrs={
            "long_name": "Depth at cell center",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
            "edges": "z_i",
        },
    )

    if include_deptho:
        deptho = np.array(
            [
                np.random.default_rng(seed).uniform(0.0, 5.0, 5),
                np.random.default_rng(seed).uniform(0.0, 15.0, 5),
                np.random.default_rng(seed).uniform(0.0, 185.0, 5),
                np.random.default_rng(seed).uniform(0.0, 1815.0, 5),
                np.random.default_rng(seed).uniform(0.0, 6185.0, 5),
            ]
        )

        if ("yh" not in dset.dims) or ("xh" not in dset.dims):
            dset = xy_fields(dset)

        dset["deptho"] = xr.DataArray(
            deptho,
            dims=({"yh": dset.yh, "xh": dset.xh}),
            attrs={
                "long_name": "Sea Floor Depth",
                "units": "m",
                "cell_methods": "area:mean yh:mean xh:mean time: point",
                "cell_measures": "area: areacello",
                "standard_name": "sea_floor_depth_below_geoid",
            },
        )

    return dset
