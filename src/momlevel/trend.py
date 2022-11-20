""" trend.py - utilities for working with trends """

import warnings

import numpy as np
import xarray as xr

__all__ = [
    "linear_detrend",
    "time_conversion_factor",
]


def time_conversion_factor(src, dst, days_per_month=30.417, days_per_year=365.0):
    """Function that returns conversion factors for common time units.

    This function returns a conversion factor for a source time unit to a
    destination time unit. Time units are specified as string inputs.

    Recognized units are "ns", "s", "min", "hr", "day", "mon" and "yr"

    For conversions involving months, an average value of 30.417 days per
    month is assumed but this value can be overridden with the `days_per_month`
    optional argument. For conversions involving years, a value of 365 days
    per year is the default but can be overridden at runtime with the
    optional `days_per_year` argument.

    Parameters
    ----------
    src : str
        Source time unit string
    dst : str
        Destination time unit string
    days_per_month : float, optional
        Days per month, by default 30.417
    days_per_year : float, optional
        Days per year, by default 365.0

    Returns
    -------
    float
        Time conversion factor
    """

    # conversion factors t to nanoseconds
    ns_from = {
        "ns": 1.0,
        "s": 1.0e9,
        "min": 1.0e9 * 60.0,
        "hr": 1.0e9 * 60.0 * 60.0,
        "day": 1.0e9 * 60.0 * 60.0 * 24.0,
        "mon": 1.0e9 * 60.0 * 60.0 * 24.0 * days_per_month,
        "yr": 1.0e9 * 60.0 * 60.0 * 24.0 * days_per_year,
    }

    # conversion factors from nanoseconds to t
    ns_to = {k: 1.0 / v for k, v in ns_from.items()}

    # validate inputs
    assert str(src) in ns_from.keys(), f"Source unit `{src}` not recognized"
    assert str(dst) in ns_to.keys(), f"Destination unit `{dst}` not recognized"

    return ns_from[src] * ns_to[dst]


def _detrend_array(arr, dim="time", order=1, mode="remove"):
    """Internal function to detrend an xarray.DataArray object"""

    # test input array to make sure it is supported
    assert isinstance(
        arr, xr.DataArray
    ), "`_detrend_array` only supports `xarray.DataArray` objects"

    # only linear detrending is supported; interface designed so this could be
    # make higher-order in the future
    assert (
        order == 1
    ), "Only linear detrending (i.e. `order=1`) is supported in this version."

    # get a clean interpolation index for the requested dimension
    interp_index = np.array(xr.core.missing.get_clean_interp_index(arr, dim))
    interp_index = xr.DataArray(interp_index, coords={dim: arr[dim]})

    # save the variable name for reassignment at the end
    varname = arr.name

    # perform the detrending and capture the slope and intercept
    ds_poly = arr.polyfit(dim, order)
    slope = ds_poly.polyfit_coefficients.sel(degree=1)
    intercept = ds_poly.polyfit_coefficients.sel(degree=0)

    # broadcast time against slope
    # slope_2, time_2 = xr.broadcast(slope, interp_index)

    # construct the fitted line
    fit_x = slope * interp_index

    if mode not in ["remove", "correct"]:
        raise ValueError(f"Unknown detrend mode '{mode}'")

    if mode == "remove":
        fit_x = fit_x + intercept

    # cast back as xarray.DataArray
    fit_x = xr.DataArray(fit_x, coords={dim: arr[dim]})

    # subtract the fitted line from the original array
    result = arr - fit_x

    # correct the name and attributes
    result.attrs = arr.attrs
    result.attrs[
        "detrend_comment"
    ] = f"detrended using momlevel (mode={mode}) with m={slope} and b={intercept}"
    result = result.rename(varname)

    return result


def linear_detrend(xobj, dim="time", order=1, mode="remove"):
    """Function to linearly detrend an xarray object

    This function performs a linear de-trending of either an xarray DataArray
    or Dataset. The function can either remove the linear mean and return
    anomalies relative to the trend, or it can correct for a linear trend
    while preserving the original magnitude of the input data.

    This function can operate on a single timeseries as well as a
    multi-dimensional array that contains a time dimension. In the case
    of a multi-dimensional array, the trends are calculated independently
    for each point.

    Parameters
    ----------
    xobj : xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
        Input xarray object
    dim : str, optional
        Name of coordinate for linear detrending, by default "time"
    order : int, optional
        Order of polynomial fit. Linear detrending defaults to "1"
    mode : str, optional
        Either "remove" the linear trend and return anomalies, or
        "correct" for a drift and retain the original magnitude of
        the input data, by default "remove"

    Returns
    -------
    xarray.core.dataarray.DataArray or xarray.core.dataset.Dataset
    """

    # case 1: input object is xarray.DataArray
    if isinstance(xobj, xr.DataArray):
        result = _detrend_array(xobj, dim=dim, order=order, mode=mode)

    # case 2: input object is xarray.Dataset
    elif isinstance(xobj, xr.Dataset):

        varlist = list(xobj.keys())

        # quick sanity check
        questionable_vars = ["time_bnds", "average_T1", "average_T2", "average_DT"]
        if any(var in varlist for var in questionable_vars):
            warnings.warn(
                "Incompatible variable detected. "
                + f"Check your dataset for the following and remove: {questionable_vars}"
            )

        # setup dictionary to hold results
        result = {}

        # iterate over variables that contain the dimension in question
        for var in varlist:
            result[var] = (
                _detrend_array(xobj[var], dim=dim, order=order, mode=mode)
                if dim in xobj[var].dims
                else xobj[var]
            )

        # convert dict back to xarray.Dataset
        result = xr.Dataset(result)

    else:
        raise TypeError("Input must be xarray.DataArray or xarray.Dataset")

    return result
