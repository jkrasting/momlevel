""" derived.py - module for calculating derived fields """

import xarray as xr
from momlevel import util


__all__ = ["calc_masso", "calc_rho", "calc_rhoga", "calc_volo"]


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


def calc_rho(thetao, so, pres, eos="Wright"):
    """Function to calculate in situ density

    This function calculates in situ density from potential temperature,
    salinity, and pressure. The equation of state can be specified with
    and available one from the momlevel.eos module.

    Parameters
    ----------
    thetao : xarray.core.dataarray.DataArray
        Sea water potential temperature in units = degC
    so : xarray.core.dataarray.DataArray
        Sea water salinity in units = 0.001
    pres : xarray.core.dataarray.DataArray
        Pressure, in units of Pa
    eos : str
        Equation of state, by default "Wright"

    Returns
    -------
    xarray.core.dataarray.DataArray
        In situ sea water density
    """

    # obtain the function object corresponding to the eos
    eos_func = util.eos_func_from_str(eos)

    rho = xr.apply_ufunc(
        eos_func,
        thetao,
        so,
        pres,
        dask="allowed",
    )

    rho.attrs = {
        "standard_name": "sea_water_density",
        "long_name": "In situ sea water density",
        "comment": f"calculated with the {eos} equation of state",
        "units": "kg m-3",
    }

    return rho


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
