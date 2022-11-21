""" trend.py - utilities for working with trends """

import warnings

import xarray as xr

__all__ = [
    "broadcast_trend",
    "calc_linear_trend",
    "linear_detrend",
    "time_conversion_factor",
]


def broadcast_trend(slope, dim_arr):
    """Function to broadcast a trend along a dimension

    This function broadcasts a trend against a dimension to obtain a
    fitted line, e.g. (m * x).

    Parameters
    ----------
    slope : xarray.core.dataarray.DataArray
        Array containing a slope. If this is a time trend, the units
        of the trend will be inferred from the array's `units` atttribute
        according to CF-convention. If the `units` attribute is missing,
        Xarray's default time units of [ns] wil be assumed
    dim_arr : xarray.core.dataarray.DataArray
        Dimension array, e.g. time axis

    Returns
    -------
    xarray.core.dataarray.DataArray
        Broadcasted array, or fitted line
    """

    # Make sure slope is a single array
    assert isinstance(slope, xr.DataArray), "Input `slope` must be a DataArray object"

    # Make sure in the input dimension array is 1-dimensional
    assert isinstance(
        dim_arr, xr.DataArray
    ), "Input `dim_arr` must be a DataArray object"
    assert len(dim_arr.dims) == 1, "Input `dim_arr` can only have one dimension"

    # Get the dimension name for reuse
    dim_name = dim_arr.dims[0]

    # Determine if we have time trend. If so, check the units and convert
    # to nanoseconds if necessary
    time_indexes = [xr.coding.cftimeindex.CFTimeIndex]
    if any([isinstance(dim_arr.indexes[dim_name], x) for x in time_indexes]):

        # Flag to throw default behavior warning
        warn_time_units = False

        # If slope/trend array has a units attribute, try to do something
        if "units" in slope.attrs.keys():
            units = slope.attrs["units"].split(" ")
            units = [x.replace("-1", "") for x in units if "-1" in x]

            # no acceptable units, issue default warning
            if len(units) == 0:
                warn_time_units = True

            # one acceptable unit found, convert if necessary
            elif len(units) == 1:
                units = units[0]
                if units != "ns":
                    factor = 1.0 / time_conversion_factor(units, "ns")
                    slope = slope * factor

            # ill-defined case
            else:
                raise ValueError(
                    f"Units attribute for slope `{slope.name}` "
                    + f"has multiple time definitions: {slope.attrs['units']}. "
                )

        # If no units attribute present, throw default behavior warning
        else:
            warn_time_units = True

        # Issue warning
        if warn_time_units:
            warnings.warn(
                "Unable to determine time unit of slope/trend. "
                + "Assuming Xarray's default nanoseconds (ns). "
                + "To fix this, ensure that the slope array has a units "
                + "attribute that describes the time units of the trend, "
                + "e.g. `m yr-1`"
            )

    # Calculate a clean interpolation index
    interp_index = xr.core.missing.get_clean_interp_index(dim_arr, dim_name)
    interp_index = xr.DataArray(interp_index, coords={dim_name: dim_arr[dim_name]})

    return slope * interp_index


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

    # save the variable name for reassignment at the end
    varname = arr.name

    # Calculate the slope and intercept
    ds_trend = calc_linear_trend(arr, dim=dim)
    slope = ds_trend[f"{varname}_slope"]
    intercept = ds_trend[f"{varname}_intercept"]

    # construct the fitted line
    fit_x = broadcast_trend(slope, arr[dim])

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


def calc_linear_trend(arr, dim="time", time_units=None):
    """Function to calculate the linear trend of a DataArray

    This function calculates the linear trend of an xarray DataArray object.
    The trend can be done along any array dimension, but this is most
    commonly time. The function calls the Xarray's `polyfit` function
    which uses NumPy's `polyfit` routine. The linear trend is defined as
    `order=1` within these functions.

    Xarray's internal clock is based on nanoseconds. The trend is returned
    in units of "ns-1" by default. Alternative time units can be specified
    and a conversion is performed. Recognized units are: "ns", "s", "min",
    "hr", "day", "mon" and "yr"

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input array
    dim : str, optional
        Dimension for detrending operation, by default "time"
    time_units : str, optional
        Result time units, see above, by default None

    Returns
    -------
    xarray.core.dataset.Dataset
        Xarray Dataset with variable _slope and _intercept arrays
    """

    # Capture variable name
    varname = arr.name

    # test input array to make sure it is supported
    assert isinstance(
        arr, xr.DataArray
    ), "`_detrend_array` only supports `xarray.DataArray` objects"

    # Xarray's internal polyfit func, which in turn calls numpy
    ds_poly = arr.polyfit(dim, 1)

    # the slope
    slope = ds_poly.polyfit_coefficients.sel(degree=1)
    slope = slope.drop_vars(["degree"])
    slope.attrs = arr.attrs
    slope.attrs["comment"] = "Slope of linear trend"
    slope = slope.rename(f"{varname}_slope")

    # the intercept
    intercept = ds_poly.polyfit_coefficients.sel(degree=0)
    intercept = intercept.drop_vars(["degree"])
    intercept.attrs = arr.attrs
    intercept.attrs["comment"] = "Y-intercept of linear trend"
    intercept = intercept.rename(f"{varname}_intercept")

    # determine slope units
    time_indexes = [xr.coding.cftimeindex.CFTimeIndex]
    if any([isinstance(arr.indexes[dim], x) for x in time_indexes]):
        time_units = "ns" if time_units is None else time_units

        # get existing units string if it exists
        if "units" in slope.attrs.keys():
            _units = slope.attrs["units"] + " "
        else:
            _units = ""
        _units = f"{_units} {time_units}-1"

        # time unit conversion factor
        factor = 1.0 / time_conversion_factor("ns", time_units)

        # apply scaling factor and modify units attribute
        slope = (slope * factor).assign_attrs(slope.attrs)
        slope.attrs["units"] = _units

    # Return a dataset with results
    dsout = xr.Dataset({f"{varname}_slope": slope, f"{varname}_intercept": intercept})

    return dsout


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
