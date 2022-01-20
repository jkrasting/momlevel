""" derived.py - module for calculating derived fields """

import numpy as np
import xarray as xr
from momlevel import util


__all__ = ["calc_alpha", "calc_dz", "calc_masso", "calc_rho", "calc_rhoga", "calc_volo"]


def calc_alpha(thetao, so, pres, eos="Wright"):
    """Function to calculate the thermal expansion coefficient (alpha)

    This function calculates the thermal expansion coefficient from
    potential temperature, salinity, and pressure. The equation of state
    can be specified with and available one from the momlevel.eos module.

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
        Thermal expansion coefficient
    """

    # obtain the function object corresponding to the eos
    eos_func = util.eos_func_from_str(eos, func_name="alpha")

    alpha = xr.apply_ufunc(
        eos_func,
        thetao,
        so,
        pres,
        dask="allowed",
    )

    alpha.attrs = {
        "long_name": "Thermal expansion coefficient",
        "comment": f"calculated with the {eos} equation of state",
        "units": "degC-1",
    }

    return alpha


def calc_beta(thetao, so, pres, eos="Wright"):
    """Function to calculate the haline contraction coefficient (beta)

    This function calculates the haline contraction coefficient from
    potential temperature, salinity, and pressure. The equation of state
    can be specified with and available one from the momlevel.eos module.

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
        Thermal expansion coefficient
    """

    # obtain the function object corresponding to the eos
    eos_func = util.eos_func_from_str(eos, func_name="beta")

    beta = xr.apply_ufunc(
        eos_func,
        thetao,
        so,
        pres,
        dask="allowed",
    )

    beta.attrs = {
        "long_name": "Haline contraction coefficient",
        "comment": f"calculated with the {eos} equation of state",
        "units": "PSU-1",
    }

    return beta


def calc_dz(levels, interfaces, depth, fraction=False):
    """Function to calculate dz that accounts for partial bottom cells

    This function uses the 2-dimensional bathymetry and the vertical
    coordinate levels and interfaces to calculate a 3-dimensional
    dz field that properly accounts for partial bottom cells.

    Parameters
    ----------
    levels : xarray.core.dataarray.DataArray
        Vertical coordinate cell centers (1-dimensional)
    interfaces : xarray.core.dataarray.DataArray
        Vertical coordinate cell interfaces (1-dimensional)
    depth : xarray.core.dataarray.DataArray
        Bathymetry field in same units as coordinate (2-dimensional)
    fraction : bool
        If True, return fraction of cell. If False, return raw dz,
        by default False

    Returns
    -------
    xarray.core.dataarray.DataArray
        dz (3-dimensional)
    """

    # check that all values are positive
    assert bool(
        np.all(depth.fillna(0.0) >= 0)
    ), "Depth values must all be positive-definite"
    assert bool(
        np.all(levels >= 0)
    ), "Vertical coordinate levels must all be positive-definite"
    assert bool(
        np.all(interfaces >= 0)
    ), "Vertical coordinate interfaces must all be positive-definite"

    # fill missing values with zero
    depth = depth.fillna(0.0)

    # broadcast to common dimensions
    ztop = xr.DataArray(interfaces[0:-1].values, coords=levels.coords)
    _, ztop = xr.broadcast(depth, ztop)
    zbot = xr.DataArray(interfaces[1::].values, coords=levels.coords)
    _, zbot = xr.broadcast(depth, zbot)
    depth = depth.broadcast_like(levels)

    # get pure dz
    dz_field = zbot - ztop

    # calculate partial cell
    part = depth - ztop
    part = xr.where(part < 0.0, 0.0, part)

    result = np.minimum(part, dz_field)

    if fraction:
        _dz_field = xr.where(dz_field == 0, np.nan, dz_field)
        _dz_part = xr.where(result == 0, np.nan, result)
        result = _dz_part / _dz_field

    return result


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
