""" tidegauge.py - tide gauge tools for momlevel """

import pandas as pd
import pkg_resources as pkgr
import xarray as xr
from momlevel.util import geolocate_points
from momlevel.util import tile_nominal_coords
from momlevel.util import validate_tidegauge_data

__all__ = ["extract_tidegauge"]


def extract_point(arr, row):
    """Function to extract point from an array

    This function takes an individual row from the output of
    util.geolocate_points and used the information to extract
    that location from the array.

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input data array
    row : pandas.core.series.Series
        Row from DataFrame of mapped locations. This row must
        include `name`, `dims`, and `dim_vals`

    Returns
    -------
    xarray.core.dataarray.DataArray
    """
    return xr.DataArray(
        arr.sel(**dict(zip(row["dims"], row["dim_vals"]))),
        name=row["name"],
        attrs={**arr.attrs, **dict(row)},
    ).reset_coords(drop=True)


def extract_tidegauge(
    arr, xcoord="geolon", ycoord="geolat", df_loc=None, mask=None, threshold=None
):
    """Function to extract tide gauge locations from an input array

    This function accepts and xarray.DataArray and returns a new Dataset object
    with individual variables corresponding to the requested locations. The deafult
    behavior is to use a set of US-based tide gauge locations that is included with
    this package but can be overridden by supplying a separate DataFrame.

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input data array
    xcoord : xarray.core.dataarray.DataArray or str
        x-coordinate name or object
    ycoord : xarray.core.dataarray.DataArray or str
        y-coordinate name or object
    df_loc : None, optional
        DataFrame object containing rows of tide gauge locations
    mask : xarray.core.dataarray.DataArray, optional
        Wet mask on model grid (1=ocean, 0=land)
    threshold : float, optional
        Filter locations that are insufficiently close to a model
        grid point. A value of 1 to 1.5 of the model's nominal
        resolution is a suggested value, by default None

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset containing individual DataArrays for each requested location
    """

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

    # Check that dimensions are the same for x/y coords
    _xdims = tuple(_xcoord.dims)
    _ydims = tuple(_xcoord.dims)
    assert _xdims == _ydims

    # Make sure mask does not have missing values
    mask = mask.fillna(0.0) if mask is not None else xr.ones_like(_xcoord)
    if mask.name != "mask":
        mask = mask.rename("mask")

    # Create pandas.DataFrame of model coordinate info
    df_model = pd.concat(
        [
            _xcoord.to_dataframe(),
            _ycoord.to_dataframe(),
            mask.to_dataframe(),
        ],
        axis=1,
    )

    # Get pd.DataFrame of target locations. This DataFrame must contain columns
    # named `name`, `lat`, and `lon`
    if df_loc is None:
        df_loc = pd.read_csv(
            pkgr.resource_filename("momlevel", "resources/us_tide_gauges.csv")
        )
        df_loc = df_loc.rename(columns={"PSMSL_site": "name"})

    # Call the geolocate function
    df_mapped = geolocate_points(
        df_model, df_loc, threshold=threshold, model_coords=(_ycoord.name, _xcoord.name)
    )

    # Add dim names back into data frame
    df_mapped["dims"] = [_xdims] * len(df_mapped)

    # Subset the original array for each valid location
    results = xr.Dataset(
        {row["name"]: extract_point(arr, row) for index, row in df_mapped.iterrows()}
    )

    return results
