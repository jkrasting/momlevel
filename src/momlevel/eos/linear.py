""" linear.py -- Linear Equation of State """

__all__ = [
    "density",
    "drho_dtemp",
    "drho_dsal",
    "alpha",
    "beta",
]

import numpy as np

# In a linear equation of state, the density derivatives are typically not
# calculated and are fixed parameters.

# Global reference density in kg m-3
RHO_REF = 1035.0
# Density of seawater a T=0, S=0 in kg m-3
RHO_T0_S0 = 1000.0
# Partial derivative of density with temperature in kg m-3 K-1
DRHO_DT = -0.2
# Partial derivative of density with salinity in kg m-3 PSU-1
DRHO_DS = 0.8


def density(T, S, p=None, rho_ref=None):
    """Calculate in-situ density based a linear equation of state.

    Note than in the case of a linear EOS, potential density
    is undefined as the dependence on pressure does not exist.
    In this case, potential density is essentially the same as the
    in-situ density.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa. Note that
        pressure is not used in a linear EOS. It is
        included here as an optional argument to maintain
        interfaces with other EOS implementations.
        Optional, by default None.
    rho_ref : np.float
        A reference density in kg m-3. Optional, by default None

    Returns
    -------
    numpy.ndarray
        Sea water in-situ density in kg m-3
    """

    rho = RHO_T0_S0 if rho_ref is None else (RHO_T0_S0 - rho_ref)
    rho = rho + ((DRHO_DT * T) + (DRHO_DS * S))

    return rho


def drho_dtemp(T=None, S=None, p=None):
    """Return density derivative with respect to potential temperature.

    This function returns the partial derivative of density with temperature
    for the linear equation of state. Note that temperature, salinity, and
    pressure have no effect as the density derivative is constant, but they
    are included to maintain interfaces with the other EOS implentations.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C. Optional, by default None.
    S : numpy.ndarray
        Sea water practical salinity in PSU. Optional, by default None.
    p : numpy.ndarray
        Sea water absolute pressure in Pa. Optional, by default None.

    Returns
    -------
    numpy.float
        Density derivative with respect to temperature in kg m-3 deg C -1
    """

    return DRHO_DT


def drho_dsal(T=None, S=None, p=None):
    """Return density derivative with respect to practical salinity.

    This function returns the partial derivative of density with salinity
    for the linear equation of state. Note that temperature, salinity, and
    pressure have no effect as the density derivative is constant, but they
    are included to maintain interfaces with the other EOS implentations.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C. Optional, by default None.
    S : numpy.ndarray
        Sea water practical salinity in PSU. Optional, by default None.
    p : numpy.ndarray
        Sea water absolute pressure in Pa. Optional, by default None.

    Returns
    -------
    numpy.float
        Density derivative with respect to salinity in kg m-3 PSU-1
    """

    return DRHO_DS


def alpha(T, S, p):
    """Calculate thermal expansion coefficient (alpha)

    This implementation is for the linear equation of state.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa. Note that
        pressure is not used in a linear EOS. It is
        included here as an optional argument to maintain
        interfaces with other EOS implementations.
        Optional, by default None.

    Returns
    -------
    numpy.ndarray
        Thermal expansion coefficient in deg C -1
    """
    return -1.0 * (np.full_like(T, fill_value=DRHO_DT) / density(T, S, p))


def beta(T, S, p):
    """Calculate haline contraction coefficient (beta)

    This implementation is for the linear equation of state.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa. Note that
        pressure is not used in a linear EOS. It is
        included here as an optional argument to maintain
        interfaces with other EOS implementations.
        Optional, by default None.

    Returns
    -------
    numpy.ndarray
        Thermal expansion coefficient in deg C -1
    """
    return np.full_like(T, fill_value=DRHO_DS) / density(T, S, p)
