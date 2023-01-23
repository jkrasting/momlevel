""" test_data - module for generating test data """

import numpy as np
import xarray as xr

from .horizontal import xy_fields

__all__ = [
    "zlevel_fields",
]


def zlevel_fields(dset=None, include_deptho=True):
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

    deptho = np.array(
        [
            np.random.uniform(0.0, 5.0, 5),
            np.random.uniform(0.0, 15.0, 5),
            np.random.uniform(0.0, 185.0, 5),
            np.random.uniform(0.0, 1815.0, 5),
            np.random.uniform(0.0, 6185.0, 5),
        ]
    )

    if include_deptho:
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
