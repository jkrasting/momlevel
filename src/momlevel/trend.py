""" trend.py - utilities for working with trends """

import warnings
import cftime
import numpy as np
import xarray as xr

from xarray.core.missing import get_clean_interp_index

__all__ = [
    "broadcast_trend",
    "calc_linear_trend",
    "linear_detrend",
    "time_conversion_factor",
    "seasonal_model",
    "deseason",
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
    interp_index = get_clean_interp_index(dim_arr, dim_name)
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


def seasonal_model(da_timeseries, tcoord="time", return_model=False):
    """Function to calculate a seasonal cycle in a time series
    This function creates a modelled time series that includes
    a linear trend and annual and semi-annual harmonics

    f(time) = b + m * time + c1 * sin(2*pi*time/year) + ...
        c2 * cos(2*pi*time/year) + c3 * sin(4*pi*time/year) + ...
        c4 * cos(4*pi*time/year) + residual

    Parameters
    ----------
    da_timeseries : xarray.core.dataarray.DataArray
        A time series in a DataArray format, which can be of
        arbitrary dimensionality
    tcoord : str, optional
        Name of the time coordinate, if present, by default "time"
    Returns
    -------
    residuals : xarray.core.dataarray.DataArray
        Residuals from modelled fit
    seasonal_model : xarray.core.dataarray.DataArray
        Modelled seasonal cycle of time series (returned only if return_model=True)
    """
    # PREPROCESSING
    # Here we find the non-time coordinates for our dataset and create a hashable
    # so that the model can be expanded to an arbitrary number of dimensions
    # If coordinates are empty we drop them
    da_timeseries = da_timeseries.reset_coords(drop=True)
    coords = [x for x in da_timeseries.coords if x != tcoord]
    coords = tuple(coords)

    coords_dict = {}
    for coord_name in coords:
        coords_dict[coord_name] = da_timeseries[f"{coord_name}"]
    hashable_coords = {key: tuple(val.values) for key, val in coords_dict.items()}

    # From here we use the same code provided by John, extended to multiple dimensions
    time_dec = (
        da_timeseries[tcoord].dt.year
        + (da_timeseries[tcoord].dt.dayofyear - 1 + da_timeseries[tcoord].dt.hour / 24)
        / 365
    )

    model = np.array(
        [np.ones(len(time_dec))]
        + [time_dec - np.mean(time_dec)]
        + [np.sin(2 * np.pi * time_dec)]
        + [np.cos(2 * np.pi * time_dec)]
        + [np.sin(4 * np.pi * time_dec)]
        + [np.cos(4 * np.pi * time_dec)]
    )

    pmodel = np.linalg.pinv(model)

    model_da = xr.DataArray(
        model,
        dims=["coeff", "time"],
        coords={"coeff": np.arange(1, 7, 1), "time": da_timeseries[tcoord]},
    )
    model_da = model_da.expand_dims(dim=hashable_coords)

    pmodel_da = xr.DataArray(
        pmodel,
        dims=["time", "coeff"],
        coords={"coeff": np.arange(1, 7, 1), "time": da_timeseries[tcoord]},
    )
    pmodel_da = pmodel_da.expand_dims(dim=hashable_coords)

    mcoeff = pmodel_da.dot(da_timeseries, dims="time")

    smodel = model_da.dot(mcoeff, dims="coeff")
    residuals = da_timeseries - smodel

    if "standard_name" in da_timeseries.attrs.keys():
        _standard_name_m = da_timeseries.attrs["standard_name"] + "_smodel"
        _standard_name_r = da_timeseries.attrs["standard_name"] + "_sresid"
    else:
        _standard_name_m = "smodel"
        _standard_name_r = "sresid"

    if "long_name" in da_timeseries.attrs.keys():
        _long_name_m = "Seasonal model, " + da_timeseries.attrs["long_name"]
        _long_name_r = "Seasonal residuals, " + da_timeseries.attrs["long_name"]
    else:
        _long_name_m = "Seasonal model"
        _long_name_r = "Seasonal residuals"

    if "units" in da_timeseries.attrs.keys():
        _units = da_timeseries.attrs["units"]
    else:
        _units = ""

    smodel.attrs["standard_name"] = _standard_name_m
    smodel.attrs["long_name"] = _long_name_m
    smodel.attrs["units"] = _units

    residuals.attrs["standard_name"] = _standard_name_r
    residuals.attrs["long_name"] = _long_name_r
    residuals.attrs["units"] = _units
    if return_model:
        return smodel, residuals
    return residuals


def seasonal_cycle_model(ts, daysinyear=365.0, tdim="time"):
    """
    Fits a seasonal cycle model to a given 1D time series.

    This function models the input time series `ts` as the sum of a linear trend,
    annual and semi-annual harmonics, and a residual term. It calculates fitted
    coefficients for the model parameters and returns them along with the residuals
    and the modeled time series.

    Parameters
    ----------
    ts : numpy.ndarray
        Input 1D time series to which the seasonal cycle model is fitted.
    daysinyear : int or float, optional
        Number of days in a year used to normalize the time series. Defaults to 365.0.
    tdim : str, optional
        Dimension label for time, not explicitly used in the computation. Defaults to
        "time".

    Returns
    -------
    mcoeff : numpy.ndarray
        Array of shape (6,) containing the fitted coefficients for the model:
        [a0, a1, b1, b2, c1, c2], where:
        - a0: intercept,
        - a1: linear trend coefficient,
        - b1, b2: coefficients for the annual harmonic,
        - c1, c2: coefficients for the semi-annual harmonic.
    residuals : numpy.ndarray
        Array of the same shape as `ts`, representing the difference between the
        input time series and the modeled series.
    smodel : numpy.ndarray
        Array of the same shape as `ts`, containing the modeled time series based on
        the seasonal cycle model.
    """

    time_length = ts.shape[0]
    if isinstance(daysinyear, float) or isinstance(daysinyear, int):
        time_dec = np.arange(time_length) / 365.0
    else:
        time_dec = np.arange(time_length) / daysinyear

    assert len(ts) == len(
        time_dec
    ), f"Chunk timeseries len is {len(ts)} but daysinyear is {len(time_dec)}"

    # Construct the model matrix
    model = np.array(
        [
            np.ones(time_length),
            time_dec - np.mean(time_dec),
            np.sin(2 * np.pi * time_dec),
            np.cos(2 * np.pi * time_dec),
            np.sin(4 * np.pi * time_dec),
            np.cos(4 * np.pi * time_dec),
        ]
    )

    # Compute pseudoinverse
    pmodel = np.linalg.pinv(model)

    # Coefficients
    mcoeff = np.dot(ts, pmodel)

    # Seasonal/trend model
    smodel = np.dot(mcoeff, model)

    # Residuals
    residuals = ts - smodel

    return mcoeff, residuals, smodel


def _detrend_deseason_chunk(chunk, axis=-1, daysinyear=365, output_format="residuals"):
    """
    Process a chunk of data and remove trends and seasonal patterns. The output could
    be the detrended residuals, the seasonal model, or the model coefficients,
    based on the specified output format.

    Parameters
    ----------
    chunk : ndarray
        Multidimensional data array where the detrending and deseasonalizing
        operations will take place. The input should follow the axis distribution
        specified by the `axis` parameter.

    axis : int, default=-1
        The axis along which the operations (trend and seasonal cycle removal)
        are applied. Typically, this corresponds to the time dimension.

    daysinyear : int, default=365
        Number of days in a year used to compute the seasonal cycle. This parameter
        should match the temporal resolution of the data (e.g., if using a 365-day
        (noleap) calendar, `daysinyear` should be set to 365).

    output_format : str, default="residuals"
        Specifies the output type. Can be one of:
        - "residuals": Returns detrended and deseasonalized residuals.
        - "model": Returns the seasonal and trend model itself.
        - "coeff": Returns the coefficients of the model fit instead.

    Returns
    -------
    ndarray or None
        Depending on the `output_format`, the output is a numpy array containing
        residuals, seasonal/trend model, or model coefficients. The shape of the
        result depends on the input and output format:
        - "residuals" or "model": Same shape as the input `chunk`.
        - "coeff": If `chunk` is 3D or higher, appends an additional dimension for
          coefficients (e.g., shape becomes `(chunk.shape[:-1], 6)`).

    Raises
    ------
    ValueError
        If `output_format` is not one of ["residuals", "model", "coeff"].
    """

    def func(ts, daysinyear=365, output_format="residuals"):
        mcoeff, residuals, smodel = seasonal_cycle_model(ts, daysinyear=daysinyear)

        if output_format == "residuals":
            result = residuals
        elif output_format == "model":
            result = smodel
        elif output_format == "coeff":
            # For coefficients, just return the coefficients themselves
            result = mcoeff
        else:
            raise ValueError(f"output_format {output_format} not recognized")

        return result

    # For coefficients, we need special handling
    if output_format == "coeff":
        # Reshape the chunk to 2D: (spatial_points, time)
        original_shape = chunk.shape
        n_spatial_points = np.prod(original_shape[:-1])
        chunk_2d = chunk.reshape(n_spatial_points, original_shape[-1])

        # Apply the function to each spatial point
        results = np.array(
            [
                func(ts, daysinyear=daysinyear, output_format=output_format)
                for ts in chunk_2d
            ]
        )

        # Reshape back to original spatial dimensions plus coefficients
        return results.reshape(original_shape[:-1] + (6,))
    else:
        return np.apply_along_axis(
            func, axis, chunk, daysinyear=daysinyear, output_format=output_format
        )


def _detrend_deseason_lazy(arr, daysinyear=365, output_format="residuals"):
    """
    Remove trend and seasonal components from time-series data in a lazy computation manner.

    This function applies a detrending and deseasoning operation to an input time-series
    dataset using lazy computation, which is suitable for out-of-core operations on
    chunked arrays. The time dimension is processed to remove linear trends and seasonality.
    The resulting dataset can either contain residuals or the coefficients of the linear
    model used for detrending and deseasoning.

    Parameters
    ----------
    arr : dask.array.Array
        The input time-series dataset, where the last dimension is assumed to be time.
    daysinyear : int, optional
        The number of days in a complete year to account for seasonal periodicity. Defaults
        to 365.
    output_format : str, optional
        Determines the format of the output. If "residuals," the function returns the
        deseasoned and detrended residuals. If "coeff," the function returns the coefficients
        of the detrending and deseasoning model instead.

    Returns
    -------
    dask.array.Array
        A new dask array with the same type as the input where the detrending and
        deseasoning operation has been applied. The output dimensions depend on the value
        of the `output_format` parameter:
        - If output_format is "residuals," the dimensions of the output array will match
          the input.
        - If output_format is "coeff," the last dimension will be replaced with a length-6
          dimension for the coefficients.

    Raises
    ------
    ValueError
        Raised if the `output_format` specified is not "residuals" or "coeff".
    """

    nchunks = len(arr.chunks)
    depth = {nchunks - 1: 0}

    # For coefficients, we need to adjust the output chunks
    if output_format == "coeff":
        # The output will have 6 coefficients instead of the time dimension
        new_chunks = list(arr.chunks[:-1])  # All chunks except time
        new_chunks.append((6,))  # Add chunk for coefficients
        meta = np.array([], dtype=arr.dtype)
    else:
        new_chunks = arr.chunks
        meta = None

    return arr.map_overlap(
        _detrend_deseason_chunk,
        depth=depth,
        boundary="none",
        dtype=arr.dtype,
        chunks=new_chunks,
        meta=meta,
        daysinyear=daysinyear,
        output_format=output_format,
    )


def deseason(arr, tdim="time", output_format="residuals"):
    """
    Removes seasonal and linear trends from a time-series dataset.

    This function applies detrending and deseasonalizing operations on
    an `xarray.DataArray` object. It can return residuals, the fitted
    model, or the model coefficients based on the specified `output_format`.
    The input array is processed along a specified time dimension, and the
    output array maintains the same shape as the input in all configurations
    except for coefficient mode.

    The model equation is:

    .. math::
        y(t) = a_0 + a_1t + b_1\sin(2\pi t) + b_2\cos(2\pi t) + c_1\sin(4\pi t) + c_2\cos(4\pi t) + \epsilon(t)

    where:
    - :math:`t` is time in decimal years
    - :math:`a_0` is the constant term (mean)
    - :math:`a_1` is the linear trend coefficient
    - :math:`b_1, b_2` are the annual cycle coefficients
    - :math:`c_1, c_2` are the semi-annual cycle coefficients
    - :math:`\epsilon(t)` represents the residuals

    The annual cycle amplitude and phase can be computed from the coefficients as:
    .. math::
        A_{annual} = \sqrt{b_1^2 + b_2^2}
        \phi_{annual} = \arctan2(b_2, b_1)

    Parameters
    ----------
    arr : xr.DataArray
        Input data as an `xarray.DataArray` object. The data should have a
        time dimension specified by `tdim`. If no chunking is applied to
        the array along the time dimension, it will be rechunked.
    tdim : str, default="time"
        The name of the time dimension in `arr` along which the
        computation will be performed. The default value is "time".
    output_format : {"residuals", "model", "coeff"}, default="residuals"
        Specifies the type of output to return:
        - "residuals": Returns the daily residuals after detrending and
          removing seasonal signals.
        - "model": Returns the linear trend and seasonal cycle model.
        - "coeff": Returns the polynomial coefficients of the fitted
          seasonal model.

    Returns
    -------
    xr.DataArray
        The processed `xarray.DataArray` after removing trends and
        seasonal components. The exact contents will depend on the
        selected `output_format`.

    """

    # Apply the detrending and deseasonalizing, returning residuals.
    # The output array will have the same dimensions as `da`.

    # this function only works on xarray DataArray objects
    assert isinstance(arr, xr.DataArray), "Input must be an xarray DataArray"

    # store array attributes for reassignment at the end
    attrs = arr.attrs

    # check that the specified time dimension exists
    core_dims = list(arr.dims)
    assert tdim in core_dims, (
        f"Core dim {tdim} not found. " + "Specify alternate with tdim option."
    )

    # prepare chunking structure for input array
    keep_lazy = True
    chunks = arr.chunks
    if chunks is None:
        # chunk along the time dimension to convert to dask array
        arr = arr.chunk({tdim: len(arr[tdim])})
        keep_lazy = False
    else:
        # the dask array must have a continuous chunk along the time dimension, otherwise
        # the model will fit each chunk independently and not the entire array

        # determine where the time index sits in the dimension ordering
        tdim_position = core_dims.index(tdim)

        # rechunk the time dimension, if needed
        if len(chunks[tdim_position]) != 1:
            arr = arr.chunk({tdim: len(arr[tdim])})

    # determine an array of the days in a year
    daysinyear = np.array(
        [
            366 if x is True else 365
            for x in [
                cftime.is_leap_year(x.year, x.calendar) for x in list(arr.time.values)
            ]
        ]
    )

    # reorder the dimensions so that the time dimension is the last in the list
    if tdim in core_dims:
        core_dims.remove(tdim)
        core_dims.append(tdim)
    else:
        raise ValueError(
            f"Core dimension '{tdim}' not found in array. Specify alternate with tdim option."
        )

    # set output core dims based on the mode
    if output_format in ["residuals", "model"]:
        output_dims = core_dims
    elif output_format == "coeff":
        # Replace time dimension with coefficient dimension
        output_dims = core_dims[:-1] + ["coeff"]
        coeff_coords = [
            "constant",
            "trend",
            "sin_annual",
            "cos_annual",
            "sin_semiannual",
            "cos_semiannual",
        ]
    else:
        raise ValueError(f"output_format {output_format} not recognized")

    # cast output dimensions as a tuple
    output_dims = tuple(output_dims)

    # call the core calculation routines
    result = xr.apply_ufunc(
        _detrend_deseason_lazy,
        arr,
        kwargs={"daysinyear": daysinyear, "output_format": output_format},
        input_core_dims=[core_dims],
        output_core_dims=[output_dims],
        dask="allowed",
    )

    # transpose dimensions of the result
    if output_format == "coeff":
        result = result.assign_coords(coeff=coeff_coords)
        result = result.transpose("coeff", ...)
    else:
        result = result.transpose("time", ...)

    # Calculate if original input array was not a dask array
    if keep_lazy is False:
        result = result.load()

    # update attribute values based on mode
    attrs.pop("standard_name", None)
    if output_format == "residuals":
        if "long_name" in attrs.keys():
            attrs["long_name"] = (
                attrs["long_name"] + " residuals from detrending and deseasonalizing"
            )
        attrs["processing"] = "Residuals from detrending and deseasonalizing"
    elif output_format == "model":
        if "long_name" in attrs.keys():
            attrs["long_name"] = (
                attrs["long_name"] + " model of linear trend and seasonal cycle"
            )
        attrs["processing"] = "Model of linear trend and seasonal cycle"
    elif output_format == "coeff":
        if "long_name" in attrs.keys():
            attrs["long_name"] = (
                attrs["long_name"] + " seasonal model polynomial coefficients"
            )
        attrs["processing"] = "Seasonal model polynomial coefficients"
        attrs.pop("units", None)

    # reassign attributes
    result.attrs = attrs

    return result
