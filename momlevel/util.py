""" util.py - generic utilities for momlevel """

import warnings
import numpy as np
from momlevel import eos


__all__ = ["default_coords", "validate_areacello", "validate_dataset"]


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
