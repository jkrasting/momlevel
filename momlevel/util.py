""" util.py - generic utilities for momlevel """

import warnings
import xgcm
import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree
from momlevel import eos

__all__ = [
    "annual_average",
    "default_coords",
    "get_xgcm_grid",
    "geolocate_points",
    "tile_nominal_coords",
    "validate_areacello",
    "validate_dataset",
    "validate_tidegauge_data",
]


def annual_average(xobj, tcoord="time"):
    """Function to calculate annual averages

    This function calculates the annual average of the supplied xarray object.
    The average is weighted by the number of days in the month, as inferred
    from the calendar attributes of the time coordinate objects. Non-numeric
    variables are skipped.

    Parameters
    ----------
    xobj : xarray.core.dataset.Dataset or xarray.core.dataarray.DataArray
        Input xarray object
    tcoord : str, optional
        Name of time coordinate, by default "time"

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    calendar = xobj[tcoord].values[0].calendar

    dim_coords = set(xobj.dims).union(set(xobj.coords))

    if isinstance(xobj, xr.core.dataset.Dataset):
        variables = set(xobj.variables) - dim_coords
        _xobj = xr.Dataset()
        for var in variables:
            if xobj[var].dtype not in ["object", "timedelta64[ns]"]:
                _xobj[var] = xobj[var]
    else:
        _xobj = xobj

    groups = _xobj.groupby(f"{tcoord}.year")

    _annuals = []
    for grp in sorted(dict(groups).keys()):
        assert len(groups[grp][tcoord]) == 12
        _annuals.append(
            groups[grp].weighted(groups[grp][tcoord].dt.days_in_month).mean(tcoord)
        )

    result = xr.concat(_annuals, dim="time")
    result = result.transpose("time", ...)

    def _find_ann_midpoint(year, calendar=calendar):
        """finds the midpoint of the year along the time dimension"""
        bounds = xr.cftime_range(
            f"{year}-01-01", freq="YS", periods=2, calendar=calendar
        )
        return bounds[0] + ((bounds[1] - bounds[0]) / 2)

    new_time_axis = [
        _find_ann_midpoint(str(x).zfill(4), calendar=calendar)
        for x in dict(groups).keys()
    ]

    result = result.assign_coords({"time": new_time_axis})

    if isinstance(xobj, xr.core.dataset.Dataset):
        for var in list(result.variables):
            result[var].attrs = xobj[var].attrs if var in list(xobj.variables) else {}

    result.attrs = xobj.attrs

    return result


def default_coords(coord_names=None):
    """Function to set the default coordinate names

    This function reads the coordinate names dictionary and returns
    either the specified values or the default values if missing from
    coord_names.

    Parameters
    ----------
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}. Coordinate
        bounds are noted by appending "bounds" to each key, i.e. "zbounds"

    Returns
    -------
    Tuple
        Default coordinate names
    """
    # abstracting these coordinates in case they need to be promoted to kwargs
    coord_names = {} if coord_names is None else coord_names
    assert isinstance(coord_names, dict), "Coordinate mapping must be a dictionary."
    zcoord = coord_names["z"] if "z" in coord_names.keys() else "z_l"
    zbounds = coord_names["zbounds"] if "zbounds" in coord_names.keys() else "z_i"
    tcoord = coord_names["t"] if "t" in coord_names.keys() else "time"
    return (tcoord, zcoord, zbounds)


def eos_func_from_str(eos_str, func_name="density"):
    """Function to resolve equation of state function

    This function takes the name of an equation of state in string
    format and returns the corresponding function object.

    Parameters
    ----------
    eos_str : str
        Equation of state

    Returns
    -------
    function
    """

    assert isinstance(eos_str, str), "Expecting string for equation of state"
    eos_str = eos_str.lower()
    avail_eos = list(eos.__dict__.keys())
    if eos_str not in avail_eos:
        raise ValueError(f"Unknown equation of state: {eos_str}")

    return eos.__dict__[eos_str].__dict__[func_name]


def geolocate_points(
    df_model,
    df_locs,
    threshold=None,
    model_coords=("geolat", "geolon"),
    rad_earth=6.378e03,
    loc_coords=("lat", "lon"),
    apply_mask=True,
):
    """Function to map a set of real-world locations to model grid points

    This function compares two Pandas DataFrame objects to map a set of
    real-world locations to their nearest model grid points. The function
    generates a ball tree for computational efficiency and uses a
    haversine, or great-circle, distance metric. An optional distance
    threshold may be specified to filter out locations that are too far
    away from a model grid point.

    Parameters
    ----------
    df_model : pandas.core.frame.DataFrame
        DataFrame containing all model grid points
    df_locs : pandas.core.frame.DataFrame
        DataFrame containing real-world locations to map to model grid
    threshold : float, optional
        Filter out points that exceed this distance threshold in km,
        by default None
    model_coords : Tuple[str, str], optional
        Names of DataFrame columns corresponding to the columns that
        identify the model's latitude and longitude values,
        by default ("geolat","geolon")
    rad_earth : float, optional
        Radius of the earth, by default 6.378e03 km
    loc_coords : Tuple[str, str]
        Names of DataFrame columns corresponding to the columns that
        identify the real-world latitude and longitude values,
        by default ("lat","lon")
    apply_mask : bool, optional
        Only consider valid model points based on the values of the `mask`
        column in `df_model`.  If `mask` == 1, it is considered a valid
        point. By default, True

    Returns
    -------
    pandas.core.frame.DataFrame
        The `df_locs` DataFrame is returned with additional columns that
        map the points to the model grid

        mod_index : int
            Index value of selected point in `df_model`
        distance : float
            Physical distance in km between the location and the selected
            grid cell location
        model_coords : tuple(float,float)
            Coordinates of selected model grid cell
        dim_vals : tuple(float or int, float or int)
            Dimension values of selected model grid cell
    """

    # Expand coords from kwargs
    ycoord1, xcoord1 = model_coords
    ycoord2, xcoord2 = loc_coords

    # Make copies of dataframes to avoid overwriting
    df1 = df_model.copy()
    df2 = df_locs.copy()

    if apply_mask:
        df1 = df1[df1["mask"] == 1.0] if "mask" in df1.columns else df1

    # Convert degree coords to radians
    df1["xrad"] = np.deg2rad(df1[xcoord1])
    df1["yrad"] = np.deg2rad(df1[ycoord1])
    df2["xrad"] = np.deg2rad(df2[xcoord2])
    df2["yrad"] = np.deg2rad(df2[ycoord2])

    # Construct the ball tree using the haversine distance metric
    ball = BallTree(df1[["yrad", "xrad"]].values, metric="haversine")

    # Locate the nearest model point for each location; convert to km
    df2["distance"], df2["mod_index"] = ball.query(df2[["yrad", "xrad"]].values, k=1)
    df2["distance"] = df2["distance"] * rad_earth

    # Filter by distance if requested
    df2 = df2[df2["distance"] <= threshold] if threshold is not None else df2

    # Add model coordinates to the location dataframe
    df1 = df1.iloc[df2["mod_index"].values]
    df2["model_coords"] = list(zip(df1[ycoord1].values, df1[xcoord1].values))
    df2["dim_vals"] = list(df1.index)

    # Clean up coordinates
    df2["real_coords"] = list(zip(df2["lat"].values, df2["lon"].values))
    df2 = df2.drop(["yrad", "xrad", "lat", "lon"], axis=1)

    return df2


def get_xgcm_grid(dset, coord_dict=None, symmetric=False):
    """Function to generate xgcm grid

    This function generates an xgcm grid based on an input dataset.
    Default MOM6 coordinate names are assumed but can be overridden
    using the `coord_dict` kwarg. Symmetric grids should be identified
    by setting `symmetric=True`.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    coord_dict : dict, optional
        Dictionary of xgcm coordinate name mappings, if different from
        the MOM6 default values, by default None
    symmetric : bool
        Flag denoting symmetric grid, by default False

    Returns
    -------
    xgcm.Grid
        Grid object from xgcm
    """

    # define a dictionary of coordinate names if not provided
    if coord_dict is None:
        coord_dict = {
            "xcenter": "xh",
            "ycenter": "yh",
            "xcorner": "xq",
            "ycorner": "yq",
        }

    if symmetric:
        result = xgcm.Grid(
            dset,
            coords={
                "X": {"center": coord_dict["xcenter"], "outer": coord_dict["xcorner"]},
                "Y": {"center": coord_dict["ycenter"], "outer": coord_dict["ycorner"]},
            },
            boundary=None,
        )
    else:
        result = xgcm.Grid(
            dset,
            coords={
                "X": {"center": coord_dict["xcenter"], "right": coord_dict["xcorner"]},
                "Y": {"center": coord_dict["ycenter"], "right": coord_dict["ycorner"]},
            },
            boundary=None,
        )

    return result


def tile_nominal_coords(xcoord, ycoord, warn=True):
    """Function to convert 1-D dimensions to 2-D coordinate variables

    This function converts 1-dimensional arrays of x and y coordinates
    to 2-dimensional lon and lat variables. This function is appropriate
    for regular lat-lon grids but should not be used with ocean model
    nominal coordinates associated with irregular grids.

    Parameters
    ----------
    xcoord : xarray.core.dataarray.DataArray
        x or longitude 1D coordinate
    ycoord : xarray.core.dataarray.DataArray
        y or latitude 1D coordinate
    warn : bool, optional
        Issue warning message, by default True

    Returns
    -------
    Tuple[xarray.core.dataarray.DataArray, xarray.core.dataarray.DataArray]
        2-dimensional longitude and latitude variables
    """
    assert isinstance(xcoord, xr.DataArray), "xcoord must be xarray.DataArray"
    assert isinstance(ycoord, xr.DataArray), "ycoord must be xarray.DataArray"

    if warn:
        warnings.warn(
            "Constructing coordinates from 1-D vectors. "
            + "Make sure this is the intended behavior. "
            + "Do not use `xh`/`yh` when `geolon`/`geolat` are available"
        )

    xgrp, ygrp = np.meshgrid(xcoord, ycoord)
    _xcoord = xr.DataArray(
        xgrp,
        dims=(ycoord.name, xcoord.name),
        coords={ycoord.name: ycoord, xcoord.name: xcoord},
        name="geolon",
    )
    _ycoord = xr.DataArray(
        ygrp,
        dims=(ycoord.name, xcoord.name),
        coords={ycoord.name: ycoord, xcoord.name: xcoord},
        name="geolat",
    )

    return _xcoord, _ycoord


def validate_areacello(areacello, reference=3.6111092e14, tolerance=0.02):
    """Function to test validity of ocean cell area field

    This function tests if the sum of the ocean cell area is within a
    specified tolerance of a real-world value.

    The intent of this function is to identify gross deviations, such
    as the inadvertent use of the global surface area.

    Parameters
    ----------
    areacello : xarray.core.dataarray.DataArray
        Field containing ocean cell area
    reference : float, optional
        Sum of ocean surface area in meters, by default 3.6111092e14
    tolerance : float, optional
        Acceptable +/- tolerance range percentage, by default 0.02

    Returns
    -------
    bool
        True if areacello is within tolerance
    """
    error = (areacello.sum() - reference) / reference
    result = bool(np.abs(error) < tolerance)
    return result


def validate_dataset(dset, reference=False, strict=True, additional_vars=None):
    """Function to validate requirements of the datasets

    This function determines if a supplied dataset is either a valid
    input dataset or reference dataset. It checks for the presence
    of required fields and that they have the correct dimensionality.

    Errors are collected and reported back as a group.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset supplied for validation
    reference : bool, optional
        Flag denoting if `dset` is a reference dataset, by default False
    strict : bool, optional
        If true, errors are handled as fatal Exceptions. If false,
        warnings are issued, by default True
    additional_vars : :obj:`list`, optional
        List of additional variables to check for in the dataset

    Returns
    -------
    None
    """

    dset_varlist = list(dset.variables)
    exceptions = []

    # check that reference dset does not contain a time dimension
    if reference:
        try:
            assert "time" not in [
                x.lower for x in list(dset.coords)
            ], "Reference dataset cannot contain a time coordinate"
        except AssertionError as e:
            exceptions.append(e)

    # check for missing variables
    expected_varlist = ["thetao", "so", "volcello", "areacello"]

    # add additional variables if supplied
    if additional_vars is not None:
        additional_vars = (
            [additional_vars]
            if not isinstance(additional_vars, list)
            else additional_vars
        )
    else:
        additional_vars = []
    expected_varlist = expected_varlist + additional_vars

    reference_varlist = ["rho", "volo", "masso", "rhoga"]
    expected_varlist = (
        expected_varlist + reference_varlist if reference else expected_varlist
    )

    missing = list(set(expected_varlist) - set(dset_varlist))

    try:
        assert len(missing) == 0, f"Reference dataset is missing variables: {missing}"
    except AssertionError as e:
        exceptions.append(e)

    # check for dimensionality of 3D vars
    ranks = (3, "(z,y,x)") if reference else (4, ("t,z,y,x"))
    for var in ["thetao", "so", "volcello"]:
        if var in dset.variables:
            try:
                assert (
                    len(dset[var].dims) == ranks[0]
                ), f"Variable {var} must have exactly {ranks[0]} dimensions {ranks[1]}"
            except AssertionError as e:
                exceptions.append(e)

    # check for dimensionality of 2D vars
    for var in ["areacello", "deptho"]:
        if var in dset.variables:
            try:
                assert (
                    len(dset[var].dims) == 2
                ), f"Variable {var} must have exactly 2 dimensions (y,x)"
            except AssertionError as e:
                exceptions.append(e)

    # validate ocean cell area and make sure it is sensible
    if "areacello" in dset.variables:
        try:
            assert validate_areacello(
                dset["areacello"]
            ), "Variable `areacello` field is out of range. It may not be masked."
        except AssertionError as e:
            if not strict:
                warnings.warn(str(e))
            else:
                exceptions.append(e)

    # check for dimensionality of scalars
    if reference:
        if "rho" not in missing:
            try:
                assert (
                    len(dset["rho"].dims) == 3
                ), "Variable areacello must have exactly 3 dimensions (z,y,x)"
            except AssertionError as e:
                exceptions.append(e)

        for var in ["masso", "volo", "rhoga"]:
            if var not in missing:
                try:
                    assert len(dset[var].dims) == 0, f"Variable {var} must be a scalar"
                except AssertionError as e:
                    exceptions.append(e)

    if len(exceptions) > 0:
        for e in exceptions:
            print(e)
        raise ValueError("Errors found in dataset.")


def validate_tidegauge_data(arr, xcoord, ycoord, mask):
    """Function to validate inputs to `tidegauge.extract_tidegauge`

    This function validates the input arguments before the tide gauge
    point extraction function continues

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input DataArray object
    xcoord : xarray.core.dataarray.DataArray or str
        x-coordinate name or object
    ycoord : xarray.core.dataarray.DataArray or str
        y-coordinate name or object
    mask : xarray.core.dataarray.DataArray or None
        wet mask array
    """
    # confirm that input is xarray
    assert isinstance(
        arr, xr.DataArray
    ), "Input array must be `xarray.DataArray` instance"

    # if xcoord and ycoord are strings, check that coordinates
    # exist in the xarray object.
    _coords = list(arr.coords)

    if isinstance(xcoord, str):
        assert xcoord in _coords, f"`{xcoord}` not found in input array."
    else:
        assert isinstance(xcoord, xr.DataArray), (
            "xcoord must either be a DataArray object or a "
            + "string that references an existing coordinate"
        )

    if isinstance(ycoord, str):
        assert ycoord in _coords, f"`{ycoord}` not found in input array."
    else:
        assert isinstance(ycoord, xr.DataArray), (
            "ycoord must either be a DataArray object or a "
            + "string that references an existing coordinate"
        )

    if mask is not None:
        assert isinstance(mask, xr.DataArray), "mask be a DataArray object"
