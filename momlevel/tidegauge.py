""" tidegauge.py - tide gauge tools for momlevel """

import warnings
import numpy as np
import pandas as pd
import pkg_resources as pkgr
import xarray as xr
from momlevel.util import geolocate_points
from momlevel.util import tile_nominal_coords
from momlevel.util import validate_tidegauge_data

__all__ = ["extract_tidegauge"]


def extract_tidegauge(
    arr, xcoord="geolon", ycoord="geolat", df_loc=None, mask=None, threshold=None
):

    # Validate inputs
    validate_tidegauge_data(arr, xcoord, ycoord, mask)

    # Get coordinate arrays to work with
    _xcoord = arr[xcoord] if isinstance(xcoord, str) else xcoord
    _ycoord = arr[ycoord] if isinstance(ycoord, str) else ycoord

    # Ensure coordinates have the same shape
    assert len(_xcoord.shape) == len(
        _ycoord.shape
    ), "x and y coordinates must have the same shape"

    # Construct 2D coordinates if necessary
    if len(_xcoord.shape) == 1:
        _xcoord, _ycoord = tile_nominal_coords(_xcoord, _ycoord)

    # Make sure mask does not have missing values
    mask = mask.fillna(0.0) if mask is not None else xr.ones_like(_xcoord)
    if mask.name != "mask":
        mask = mask.rename("mask")

    # Create pandas.DataFrame of model coordinate info
    df = pd.concat(
        [
            _xcoord.to_dataframe(),
            _ycoord.to_dataframe(),
            mask.to_dataframe(),
        ],
        axis=1,
    )

    # Get pd.DataFrame of target locations. This DataFrame must contain columns
    # named `lat` and `lon`
    df_loc = (
        pd.read_csv(pkgr.resource_filename("momlevel", "resources/us_tide_gauges.csv"))
        if df_loc is None
        else df_loc
    )

    df_mapped = geolocate_points(
        df, df_loc, threshold=threshold, model_coords=(_ycoord.name, _xcoord.name)
    )

    return df_mapped
