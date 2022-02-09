""" util.py - generic utilities for momlevel """

import warnings
import xgcm
import numpy as np
import xarray as xr
from momlevel import eos


__all__ = [
    "annual_average",
    "default_coords",
    "get_xgcm_grid",
    "validate_areacello",
    "validate_dataset",
]


def annual_average(dset, tcoord="time"):
    """Function to calculate annual averages

    This function calculates the annual average of every variable contained
    in the supplied xarray Dataset. The average is weighted by the number of
    days in the month, as inferred from the calendar attributes of the time
    coordinate objects. Non-numeric variables are skipped.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    tcoord : str, optional
        Name of time coordinate, by default "time"

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    calendar = dset[tcoord].values[0].calendar

    dim_coords = set(dset.dims).union(set(dset.coords))
    variables = set(dset.variables) - dim_coords

    _dset = xr.Dataset()
    for var in variables:
        if dset[var].dtype not in ["object", "timedelta64[ns]"]:
            _dset[var] = dset[var]

    groups = _dset.groupby(f"{tcoord}.year")

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

    for var in list(result.variables):
        result[var].attrs = dset[var].attrs if var in list(dset.variables) else {}

    result.attrs = dset.attrs

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
        print("Doing symmetric")
        result = xgcm.Grid(
            dset,
            coords={
                "X": {"center": coord_dict["xcenter"], "outer": coord_dict["xcorner"]},
                "Y": {"center": coord_dict["ycenter"], "outer": coord_dict["ycorner"]},
            },
            periodic=["X"],
        )
    else:
        result = xgcm.Grid(
            dset,
            coords={
                "X": {"center": coord_dict["xcenter"], "right": coord_dict["xcorner"]},
                "Y": {"center": coord_dict["ycenter"], "right": coord_dict["ycorner"]},
            },
            periodic=["X"],
        )

    return result


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
