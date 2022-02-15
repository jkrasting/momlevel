""" derived.py - module for calculating derived fields """

import xgcm
import numpy as np
import xarray as xr
from momlevel import util


__all__ = [
    "calc_alpha",
    "calc_beta",
    "calc_dz",
    "calc_n2",
    "calc_masso",
    "calc_pv",
    "calc_rel_vort",
    "calc_rho",
    "calc_rhoga",
    "calc_volo",
]


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
        Sea water absolute pressure, in units of Pa
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
        Sea water absolute pressure, in units of Pa
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


def calc_rel_vort(dset, varname_map=None, coord_dict=None, symmetric=False):
    """Function to calculate relative vorticity

    This function calculates the vertical component of the relative vorticity
    from the u and v components of the flow.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset. Required variables are `uo`, `vo`, `dxCu`, `dyCv`,
        and `areacello_bu`
    varname_map : :obj:`dict`, optional
        Dictionary of variable mappings. Variables are renamed according to these
        mappings at the start of the routine, by default None.
    coord_dict : :obj:`dict`, optional
        Dictionary of xgcm coordinate name mappings, if different from
        the MOM6 default values, by default None
    symmetric : bool
        Flag denoting symmetric grid, by default False

    Returns
    -------
    xarray.core.dataarray.DataArray
        Ocean relative vorticity in s-1
    """

    if varname_map is None:
        varname_map = {
            "u": "uo",
            "v": "vo",
            "dx": "dxCu",
            "dy": "dyCv",
            "area": "areacello_bu",
        }

    # check that dataset contains u and v fields
    required = set(varname_map.values())
    varnames = set(dset.variables)
    missing = list(required - varnames)
    if len(missing) > 0:
        raise ValueError(f"Input dataset missing fields: {missing}")

    # get xgcm grid object
    grid = util.get_xgcm_grid(dset, coord_dict=coord_dict, symmetric=symmetric)

    relvort = (
        -grid.diff(
            dset[varname_map["u"]] * dset[varname_map["dx"]], "Y", boundary="fill"
        )
        + grid.diff(
            dset[varname_map["v"]] * dset[varname_map["dy"]], "X", boundary="fill"
        )
    ) / dset[varname_map["area"]]

    relvort.attrs = {
        "standard_name": "ocean_relative_vorticity",
        "long_name": "Ocean relative vorticity",
        "units": "s-1",
    }
    return relvort


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


def calc_n2(
    thetao, so, eos="Wright", gravity=-9.8, patm=101325.0, zcoord="z_l", interfaces=None
):
    """Function to calculate the buoyancy frequency

    This function calculates the Brunt-Väisälä frequency which is commonly used
    to evaluate stratification. The equivalent field in CMIP is `obvfsq`.

    The buoyancy frequency is calculated independent of the effects of pressure
    by calculating derivatives from locally referenced potential temperature.

    Note that this field differs from the online calculation of `obvfsq` in that
    the default behavior is to calculate the buoyancy frequency at the cell
    centers whereas the model's default behavior is calculate this quantity on
    the cell edges.

    This field can be converted to cycles per hour (CPH) via:
        np.sqrt(n2)*3600.

    Parameters
    ----------
    thetao : xarray.core.dataarray.DataArray
        Sea water potential temperature in units = degC
    so : xarray.core.dataarray.DataArray
        Sea water salinity in units = 0.001
    eos : str, optional
        Equation of state to use in calculations, by default "Wright"
    gravity : float, optional
        Gravitational acceleration constant in m s-2, by default -9.8
    patm : float or xarray.core.dataarray.DataArray
        Atmospheric pressure at the sea surface in Pa,
        by default 101325 Pa (US Standard Atmosphere)
    zcoord : str, optional
        Vertical coorindate name, by default "z_l"
    interfaces : xarray.core.dataarray.DataArray, optional
        Vertical coordinate cell interfaces, by default None
        If provided, calculation will be performed on cell edges.

    Returns
    -------
    xarray.core.dataarray.DataArray
       Brunt-Väisälä frequency, or buoyancy frequency, in s-2

    See Also
    --------
    calc_alpha : Calculates thermal expansion coefficient
    calc_beta : Calculates haline contraction coefficient
    """

    if interfaces is not None:
        _ds = xr.Dataset({"thetao": thetao, "so": so})
        grid = xgcm.Grid(_ds, coords={"Z": {"center": zcoord}}, periodic=False)
        thetao = grid.transform(thetao, "Z", interfaces, method="linear")
        so = grid.transform(so, "Z", interfaces, method="linear")
        zcoord = interfaces.name

    pres = (thetao[zcoord] * 1.0e4) + patm
    alpha = calc_alpha(thetao, so, pres, eos=eos)
    beta = calc_beta(thetao, so, pres, eos=eos)
    dtdz = thetao.differentiate(zcoord, edge_order=2)
    dsdz = so.differentiate(zcoord, edge_order=2)
    n2 = gravity * ((alpha * dtdz) - (beta * dsdz))
    n2.attrs = {
        "standard_name": "square_of_brunt_vaisala_frequency_in_sea_water",
        "long_name": "Square of seawater buoyancy frequency",
        "units": "s-2",
    }
    return n2


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


def calc_pdens(thetao, so, level=0.0, patm=101325, eos="Wright"):
    """Function to calculate potential density

    This function calculates potential density referenced to a given
    depth level (z). By default, this is the surface. The pressure associated
    with z is defined such that p = z * 1.e4.

    Parameters
    ----------
    thetao : xarray.core.dataarray.DataArray
        Sea water potential temperature in units = degC
    so : xarray.core.dataarray.DataArray
        Sea water salinity in units = 0.001
    level : float, optional
        Reference depth level in m, by default 0.
    patm : float or xarray.core.dataarray.DataArray
        Atmospheric pressure at the sea surface in Pa,
        by default 101325 Pa (US Standard Atmosphere)
    eos : str, optional
        Equation of state, by default "Wright"

    Returns
    -------
    xarray.core.dataarray.DataArray
        Sea water potential density
    """

    assert 0.0 <= level <= 7500.0, "specified level must be between 0 and 7500 m"

    # note: approximate pressure from depth
    rhopot = calc_rho(thetao, so, (level * 1.0e4) + patm, eos=eos)

    rhopot.attrs = {
        "standard_name": "sea_water_potential_density",
        "long_name": f"Sea water potential density referenced to {level} m",
        "comment": f"calculated with the {eos} equation of state",
        "units": "kg m-3",
    }

    return rhopot


def calc_pv(zeta, coriolis, n2, gravity=9.8, coord_dict=None, symmetric=False):
    """Function to calculate ocean potential vorticity

    This function calculates potential vorticity given the relative vorticity,
    Coriolis parameter, and buoyancy frequency as inputs.

    Parameters
    ----------
    zeta : xarray.core.dataarray.DataArray
        Vertical component of the relative vorticity in units = s-1
    coriolis : xarray.core.dataarray.DataArray
        Coriolis parameter grid cell corners, in units = s-1
    n2 : xarray.core.dataarray.DataArray
        Brunt-Väisälä frequency in units = s-2
    gravity : float, optional
        Gravitational acceleration constant, by default 9.8 m s-2
    coord_dict : :obj:`dict`, optional
        Dictionary of xgcm coordinate name mappings, if different from
        the MOM6 default values, by default None
    symmetric : bool
        Flag denoting symmetric grid, by default False

    Returns
    -------
    xarray.core.dataarray.DataArray
        Ocean potential vorticity in m-1 s-1

    See Also
    --------
    calc_n2 : Calculates Brunt-Väisälä (buoyancy) frequency
    calc_rel_vort : Calculate relative vorticity (zeta)
    """

    # create an internal dataset for xgcm purposes
    _dset = xr.Dataset({"zeta": zeta, "coriolis": coriolis, "n2": n2})
    grid = util.get_xgcm_grid(_dset, coord_dict=coord_dict, symmetric=symmetric)

    # interpolate N2 to the corner points
    n2 = grid.interp(n2, axis=["X", "Y"], boundary="fill")

    # calculate potential vorticity
    swpotvort = (zeta + coriolis) * (n2 / gravity)

    swpotvort.attrs = {
        "long_name": "Ocean potential vorticity",
        "units": "m-1 s-1",
    }

    return swpotvort


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
        Sea water absolute pressure, in units of Pa
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
