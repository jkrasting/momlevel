import warnings

import numpy as np
import xarray as xr
import momlevel.eos as eos


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
    result = True if np.abs(error) < tolerance else False
    return result


def calc_volo(volcello):
    """Function to calculate the total ocean volume

    This function sums the 3-dimensional volcello field and adds in
    the appropriate CF metadata.

    Parameters
    ----------
    volcello : xarray.core.dataarray.DataArray
        The volcello field as a function of z, y, and x (m3)

    Returns
    -------
    xarray.core.dataarray.DataArray
        Scalar total ocean volume (m3)
    """

    # check dimensionality
    assert len(volcello.dims) == 3, "Expecting only 3 dimensions for volcello"

    volo = volcello.sum()
    volo.attrs = {
        "standard_name": "sea_water_volume",
        "long_name": "Sea Water Volume",
        "units": "m3",
    }
    return volo


def calc_masso(rho, volcello, tcoord="time"):
    """Function to calculate the total ocean mass

    This function calculates the total ocean mass by multiplying the
    in situ density by cell volume and summing across all non-time
    coordinates and applying appropriate CF metadata.

    Parameters
    ----------
    rho : xarray.core.dataarray.DataArray
        The in situ density field in units of kg m-3
    volcello : xarray.core.dataarray.DataArray
        The volcello field in units of m3
    tcoord : str, optional
        Name of the time coordinate, if present, by default "time"

    Returns
    -------
    xarray.core.dataarray.DataArray
        Scalar total ocean mass (kg)
    """
    masso = rho * volcello
    coords = [x for x in masso.dims if x != tcoord]
    coords = tuple(coords)
    masso = masso.sum(coords)
    masso.attrs = {
        "standard_name": "sea_water_mass",
        "long_name": "Sea Water Mass",
        "units": "kg",
    }
    return masso


def calc_rhoga(masso, volo):
    """Function to calculate global average ocean density

    This function calculates the global average ocean density based
    on the total ocean mass and volume and applies appropriate
    metadata.

    Parameters
    ----------
    masso : xarray.core.dataarray.DataArray
        Total ocean mass (kg)
    volo : xarray.core.dataarray.DataArray
        Total ocean volume (m3)

    Returns
    -------
    xarray.core.dataarray.DataArray
        Scalar of global average ocean density
    """
    rhoga = masso / volo
    rhoga.attrs = {
        "long_name": "Global Average Sea Water Density",
        "units": "kg m-3",
    }
    return rhoga


def calc_rho(eos, thetao, so, pres):
    """Function to calculate in situ density

    This function calculates in situ density from potential temperature,
    salinity, and pressure. The equation of state is specified as a function
    from the momlevel.eos module, e.g. `momlevel.eos.wright`.

    Parameters
    ----------
    eos : function
        Equation of state from the momlevel.eos module
    thetao : xarray.core.dataarray.DataArray
        Sea water potential temperature in units = degC
    so : xarray.core.dataarray.DataArray
        Sea water salinity in units = 0.001
    pres : xarray.core.dataarray.DataArray
        Pressure, in units of Pa

    Returns
    -------
    xarray.core.dataarray.DataArray
        In situ sea water density
    """
    rho = xr.apply_ufunc(
        eos,
        thetao,
        so,
        pres,
        dask="allowed",
    )

    eos_name = str(eos).split(" ")[1].capitalize()
    rho.attrs = {
        "standard_name": "sea_water_density",
        "long_name": "In situ sea water density",
        "comment": f"calculated with the {eos_name} equation of state",
        "units": "kg m-3",
    }

    return rho


def default_coords(coord_names=None):
    """Function to set the default coordinate names

    This function reads the coordinate names dictionary and returns
    either the specified values or the default values if missing from
    coord_names.

    Parameters
    ----------
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}

    Returns
    -------
    Tuple
        Default coordinate names
    """
    # abstracting these coordinates in case they need to be promoted to kwargs
    coord_names = {} if coord_names is None else coord_names
    assert isinstance(coord_names, dict), "Coordinate mapping must be a dictionary."
    zcoord = coord_names["z"] if "z" in coord_names.keys() else "z_l"
    tcoord = coord_names["t"] if "t" in coord_names.keys() else "time"
    return (tcoord, zcoord)


def setup_reference_state(dset, eos, coord_names=None, time_index=0):
    """Function to generate reference dataset

    This function generates a dataset of initial reference values for
    use in all sea level calculations. Values are taken from an input
    dataset that contains thetao, so, volcello, and areacello. The
    initial in situ density is calculated along with scalar properties
    that are used in offline approximations of the steric effect from
    Boussinesq models.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset containing thetao, so, volcello, and areacello
    eos : function
        Equation of state function from momlevel.eos
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}
    time_index : int, optional
        Time index to use for reference state, by default 0 (first time index)

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset of reference values
    """

    # default coordinate names
    tcoord, zcoord = default_coords(coord_names)

    # approximate pressure from depth coordinate
    pres = dset[zcoord] * 1.0e4

    # initialize reference dataset
    reference = xr.Dataset()

    # extract requested time level and squeeze the arrays
    reference["thetao"] = (
        dset["thetao"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )
    reference["so"] = (
        dset["so"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )
    reference["volcello"] = (
        dset["volcello"].isel({tcoord: time_index}).squeeze().reset_coords(drop=True)
    )

    # calculate in situ reference density
    reference["rho"] = calc_rho(eos, reference.thetao, reference.so, pres)

    # calculate global ocean volume
    reference["volo"] = calc_volo(reference.volcello)

    # calculate global ocean mass
    reference["masso"] = calc_masso(reference.rho, reference.volcello, tcoord=tcoord)

    # calculate global average in situ density
    reference["rhoga"] = calc_rhoga(reference.masso, reference.volo)

    # copy the reference cell area
    reference["areacello"] = dset.areacello

    return reference


def validate_dataset(dset, reference=False, strict=False):
    """Function to validate requirements of the datasets

    This function determines if a supplied dataset is either a valid
    input dataset or reference dataset. It checks for the presence
    of required fields and that they have the correct dimensionality.

    Errors are collected and reported back as a group.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset supplied for validation
    reference : bool
        Flag denoting if `dset` is a reference dataset, by default False
    strict : bool
        If true, errors are handled as fatal Exceptions. If false,
        warnings are issued, by default False

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
    expected_varlist = (
        expected_varlist + ["rho", "volo", "masso", "rhoga"]
        if reference
        else expected_varlist
    )
    missing = list(set(expected_varlist) - set(dset_varlist))

    try:
        assert len(missing) == 0, f"Reference dataset is missing variables: {missing}"
    except AssertionError as e:
        exceptions.append(e)

    # check for dimensionality of 3D vars
    ranks = (3, "(z,y,x)") if reference else (4, ("t,z,y,x"))
    for var in ["thetao", "so", "volcello"]:
        if var not in missing:
            try:
                assert (
                    len(dset[var].dims) == ranks[0]
                ), f"Variable {var} must have exactly {ranks[0]} dimensions {ranks[1]}"
            except AssertionError as e:
                exceptions.append(e)

    # check for dimensionality of cell area
    if "areacello" not in missing:
        try:
            assert (
                len(dset["areacello"].dims) == 2
            ), "Variable areacello must have exactly 2 dimensions (y,x)"
        except AssertionError as e:
            exceptions.append(e)

        # validate ocean cell area and make sure it is sensible
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


def steric(
    dset,
    reference=None,
    coord_names=None,
    varname_map=None,
    equation_of_state="wright",
    variant="steric",
    domain="local",
    strict=False,
):
    """Function to calculate steric sea level change

    Calculates the steric, thermosteric, or halosteric sea level change
    and other associated quantities relative to a known reference state.

    The calculation can either be performed locally at each grid point, or
    globally. Global calculations are performed using an offline approximation
    of the steric effect commonly used for Boussinesq models.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset containing sea water potential temperature (`thetao`),
        sea water salinity (`so`), ocean grid cell volume (`volcello`), and
        ocean cell area (`areacello`)
    reference : :obj:`xarray.core.dataset.Dataset`
        Reference data set containing initial ocean state. The first time
        level of `dset` is used if not supplied, by default None
    coord_names : :obj:`dict`, optional
        Dictionary of coordinate name mappings. This should use x, y, z, and t
        as keys, e.g. {"x":"xh", "y":"yh", "z":"z_l", "t":"time"}, by default None
    varname_map : :obj:`dict`, optional
        Dictionary of variable mappings. Variables are renamed according to these
        mappings at the start of the routine, by default None.
    equation_of_state : str, optional
        Equation of state to use in calculations, by default "wright"
    variant : str, optional
        Options are "steric", "thermosteric", and "halosteric", by default "steric"
    domain : str, optional
        Options are "local" and "global", by default "local"
    strict : bool, optional
        If True, errors are handled as fatal Exceptions. If False, errors are
        passed as warnings.  By default, True

    Returns
    -------
    xarray.core.dataset.Dataset
        Results of sea level change calculation
    """

    # remap variable names, if passed
    dset = dset.rename(varname_map)

    # conduct some sanity checks on the input dataset
    validate_dataset(dset, strict=strict)

    # default coordinate names
    tcoord, zcoord = default_coords(coord_names)

    # determine the equation of state to use
    equation_of_state = eos.__dict__[equation_of_state]

    # approximate pressure from depth coordinate
    pres = dset[zcoord] * 1.0e4

    if reference is not None:
        assert isinstance(
            reference, xr.Dataset
        ), "`reference` must be an xarray Dataset"
    else:
        reference = setup_reference_state(
            dset, equation_of_state, coord_names=coord_names
        )

    # conduct some sanity checks on the reference state
    validate_dataset(reference, reference=True, strict=strict)

    # determine which fields, if any, to hold fixed
    if variant == "thermosteric":
        thetao = dset["thetao"]
        so = reference["so"]
    elif variant == "halosteric":
        thetao = reference["thetao"]
        so = dset["so"]
    elif variant == "steric":
        thetao = dset["thetao"]
        so = dset["so"]

    # calculate in situ density
    rho = calc_rho(equation_of_state, thetao, so, pres)

    # calculate the expansion coefficient for each grid cell
    if domain == "global":
        masso = calc_masso(rho, reference["volcello"], tcoord=tcoord)
        expansion_coeff = np.log(reference["rhoga"] / (masso / reference["volo"]))
    else:
        expansion_coeff = np.log(reference["rho"] / rho)
        expansion_coeff = expansion_coeff.transpose(*(tcoord, ...))
        expansion_coeff.attrs = {"long_name": "Expansion coefficient"}

    # calculate reference height and steric sea level change
    if domain == "global":
        reference_height = reference["volo"] / reference["areacello"].sum()
        sealevel = reference_height * expansion_coeff
    else:
        reference_height = reference["volcello"] / reference["areacello"]
        sealevel = (reference_height * expansion_coeff).sum(dim=zcoord)
    reference_height.attrs = {"long_name": "Reference column height", "units": "m"}

    # calculate sealevel
    sealevel = sealevel.transpose(*(tcoord, ...))
    sealevel.attrs = {
        "long_name": f"{variant.capitalize()} column height (eta)",
        "units": "m",
    }

    # return an Xarray Dataset with the results
    result = xr.Dataset()
    result["reference_thetao"] = reference.thetao
    result["reference_so"] = reference.so
    result["reference_vol"] = reference.volcello
    result["reference_rho"] = reference.rho
    result["reference_rho"] = reference.rho.where(reference.volcello.notnull())
    result["reference_height"] = reference_height.where(reference.volcello.notnull())
    result["expansion_coeff"] = expansion_coeff.where(reference.volcello.notnull())

    if domain == "global":
        result["global_reference_vol"] = reference.volo
        result["global_reference_rho"] = reference.rhoga
        result[variant] = sealevel
    else:
        result[variant] = sealevel.where(reference.volcello.isel({zcoord: 0}).notnull())

    return result


def halosteric(*args, **kwargs):
    result = steric(*args, **kwargs, variant="halosteric")
    return result


def thermosteric(*args, **kwargs):
    result = steric(*args, **kwargs, variant="thermosteric")
    return result
