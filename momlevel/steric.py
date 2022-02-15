""" steric.py - module to calculate local and global steric sea level """

import numpy as np
import xarray as xr

from momlevel.derived import calc_dz
from momlevel.derived import calc_masso
from momlevel.derived import calc_rho
from momlevel.reference import setup_reference_state
from momlevel.util import annual_average
from momlevel.util import default_coords
from momlevel.util import validate_dataset

__all__ = ["halosteric", "steric", "thermosteric"]


def steric(
    dset,
    reference=None,
    coord_names=None,
    varname_map=None,
    rhozero=1035.0,
    patm=101325.0,
    equation_of_state="Wright",
    variant="steric",
    domain="local",
    dtype="float32",
    strict=True,
    annual=False,
    verbose=False,
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
    rhozero : float, optional
        Globally constant reference density in kg m-3, by default 1035.0
    patm : float or xarray.core.dataarray.DataArray
        Atmospheric pressure at the sea surface in Pa,
        by default 101325 Pa (US Standard Atmosphere)
    equation_of_state : str, optional
        Equation of state to use in calculations, by default "Wright"
    variant : str, optional
        Options are "steric", "thermosteric", and "halosteric", by default "steric"
    domain : str, optional
        Options are "local" and "global", by default "local"
    dtype : str, optional
        Encoding data type for output, by default "float32"
    strict : bool, optional
        If True, errors are handled as fatal Exceptions. If False, errors are
        passed as warnings.  By default, True
    annual : bool, optional
        Perform annual averaging on results. By default, False
    verbose : bool, optional
        Verbose output. By default, False

    Returns
    -------
    xarray.core.dataset.Dataset
        Results of sea level change calculation
    """

    # remap variable names, if passed
    dset = dset.rename(varname_map)

    # default coordinate names
    tcoord, zcoord, zbounds = default_coords(coord_names)

    # conduct some sanity checks on the input dataset
    additional_vars = None if domain == "global" else [zbounds, "deptho"]
    validate_dataset(dset, strict=strict, additional_vars=additional_vars)

    # approximate pressure from depth coordinate
    # 1 meter of depth is approximately 1 db or 10**4 Pa and also
    # add in standard atmospheric pressure at the sea surface
    pres = (dset[zcoord] * 1.0e4) + patm

    if reference is not None:
        assert isinstance(
            reference, xr.Dataset
        ), "`reference` must be an xarray Dataset"
        if verbose:
            print("Using supplied reference state")
    else:
        reference = setup_reference_state(
            dset, patm=patm, eos=equation_of_state, coord_names=coord_names
        )
        if verbose:
            print("Generating reference state from first timestep")

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
    rho = calc_rho(thetao, so, pres, eos=equation_of_state)

    # return an Xarray Dataset with the results
    result = xr.Dataset()

    # if global, calculate reference height and steric sea level adj. for Boussinesq models
    if domain == "global":
        masso = calc_masso(rho, reference["volcello"], tcoord=tcoord)
        expansion_coeff = np.log(reference["rhoga"] / (masso / reference["volo"]))
        expansion_coeff.attrs = {"long_name": "Expansion coefficient"}
        reference_height = reference["volo"] / reference["areacello"].sum()
        reference_height.attrs = {"long_name": "Reference column height", "units": "m"}

        # global steric level approximation for Boussinesq models
        sealevel = reference_height * expansion_coeff

        result["reference_height"] = reference_height
        result["reference_height"].encoding["dtype"] = dtype

        result[variant] = sealevel

    # otherwise calculate the change in density relative to the reference
    else:
        delta_rho = xr.where(
            reference["volcello"].notnull(), rho - reference["rho"], np.nan
        )
        delta_rho = delta_rho.transpose(tcoord, ...)
        delta_rho.attrs = {
            "long_name": "change in in situ density from reference state",
            "units": "kg m-3",
        }
        result["delta_rho"] = delta_rho
        result["delta_rho"].encoding["dtype"] = dtype

        dz = calc_dz(dset[zcoord], dset[zbounds], dset["deptho"])
        sealevel = (-1.0 / rhozero) * ((dz * delta_rho).sum(zcoord))

        sealevel = sealevel.transpose(*(tcoord, ...))
        result[variant] = sealevel.where(reference.volcello.isel({zcoord: 0}).notnull())

    # fix up variable metadata
    result[variant].attrs = {
        "long_name": f"{variant.capitalize()} height adjustment",
        "units": "m",
    }

    result[variant].encoding["dtype"] = dtype

    # copy coordinate and dimenstion attributes
    for var in set(result.coords).union(result.dims):
        if var in dset.variables:
            result[var].attrs = dset[var].attrs

    if annual:
        result = annual_average(result)

    return (result, reference)


def halosteric(*args, **kwargs):
    """Wrapper for halosteric calculation"""
    result, reference = steric(*args, **kwargs, variant="halosteric")
    return (result, reference)


def thermosteric(*args, **kwargs):
    """Wrapper for thermosteric calculation"""
    result, reference = steric(*args, **kwargs, variant="thermosteric")
    return (result, reference)
