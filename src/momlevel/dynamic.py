""" dynamic.py - module for funcs related to dynamic sea level / ssh """

from momlevel.derived import calc_rho

__all__ = ["inverse_barometer"]


def inverse_barometer(tos, sos, pso, gravity=9.8, equation_of_state="Wright"):
    """Function to calculate inverse barometer height

    This function calculates the inverse barometer height given inputs of
    sea surface temperature, sea surface salinity, and a surface pressure.
    These inputs are used to calculate the surface in-situ density.

    Parameters
    ----------
    tos : xarray.core.dataarray.DataArray
        Sea surface potential temperaure in deg C
    sos : xarray.core.dataarray.DataArray
        Sea surface practical salinity in psu
    pso : xarray.core.dataarray.DataArray
        Sea surface pressure in Pa
    gravity : float, optional
        Gravitational constant, by default 9.8 m s-1
    equation_of_state : str, optional
        Equation of state to use in calculations, by default "Wright"

    Returns
    -------
    xarray.core.dataarray.DataArray
        Inverse barometer height, in m
    """

    rho_conv = calc_rho(tos, sos, pso, eos=equation_of_state)

    ibh = pso * (-1.0 / (rho_conv * gravity))

    ibh = ibh.rename("ibh")
    ibh.attrs = {"long_name": "Inverse Barometer Height", "units": "m"}

    return ibh
