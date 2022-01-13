""" steric.py - module to calculate local and global steric sea level """

import numpy as np
import xarray as xr
import momlevel.eos as eos

from momlevel.derived import calc_masso
from momlevel.derived import calc_rho

from momlevel.reference import setup_reference_state
from momlevel.util import default_coords
from momlevel.util import validate_dataset

__all__ = ["halosteric", "steric", "thermosteric"]


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
    """ Wrapper for halosteric calculation """
    result = steric(*args, **kwargs, variant="halosteric")
    return result


def thermosteric(*args, **kwargs):
    """ Wrapper for thermosteric calculation """
    result = steric(*args, **kwargs, variant="thermosteric")
    return result
