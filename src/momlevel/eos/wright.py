""" wright.py -- Wright Equation of State """

__all__ = ["density", "drho_dtemp", "drho_dsal", "alpha", "beta"]

# constants
A0 = 7.057924e-4
A1 = 3.480336e-7
A2 = -1.112733e-7
B0 = 5.790749e8
B1 = 3.516535e6
B2 = -4.002714e4
B3 = 2.084372e2
B4 = 5.944068e5
B5 = -9.643486e3
C0 = 1.704853e5
C1 = 7.904722e2
C2 = -7.984422
C3 = 5.140652e-2
C4 = -2.302158e2
C5 = -3.079464


def density(T, S, p):
    """Calculate in-situ density based on the Wright equation of state.

    Reference:
      Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa

    Returns
    -------
    numpy.ndarray
        Sea water in-situ density in kg m-3
    """

    al0 = A0 + A1 * T + A2 * S
    p0 = B0 + B4 * S + T * (B1 + T * (B2 + B3 * T) + B5 * S)
    lam = C0 + C4 * S + T * (C1 + T * (C2 + C3 * T) + C5 * S)
    I_denom = 1.0 / (lam + al0 * (p + p0))
    result = (p + p0) * I_denom

    return result


def drho_dtemp(T, S, p):
    """Calculate density derivative with respect to potential temperature.

    Reference:
      Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa

    Returns
    -------
    numpy.ndarray
        Density derivative with respect to temperature in kg m-3 deg C -1
    """

    al0 = A0 + A1 * T + A2 * S
    p0 = B0 + B4 * S + T * (B1 + T * (B2 + B3 * T) + B5 * S)
    lam = C0 + C4 * S + T * (C1 + T * (C2 + C3 * T) + C5 * S)

    I_denom2 = 1.0 / (lam + al0 * (p + p0))
    I_denom2 = I_denom2 * I_denom2
    result = I_denom2 * (
        lam * (B1 + T * (2.0 * B2 + 3.0 * B3 * T) + B5 * S)
        - (p + p0) * ((p + p0) * A1 + (C1 + T * (C2 * 2.0 + C3 * 3.0 * T) + C5 * S))
    )

    return result


def drho_dsal(T, S, p):
    """Calculate density derivative with respect to salinity.

    Reference:
      Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa

    Returns
    -------
    numpy.ndarray
        Density derivative with respect to salinity in kg m-3 PSU -1
    """

    al0 = A0 + A1 * T + A2 * S
    p0 = B0 + B4 * S + T * (B1 + T * (B2 + B3 * T) + B5 * S)
    lam = C0 + C4 * S + T * (C1 + T * (C2 + C3 * T) + C5 * S)

    I_denom2 = 1.0 / (lam + al0 * (p + p0))
    I_denom2 = I_denom2 * I_denom2
    result = I_denom2 * (
        lam * (B4 + B5 * T) - (p + p0) * ((p + p0) * A2 + (C4 + C5 * T))
    )

    return result


def alpha(T, S, p):
    """Calculate thermal expansion coefficient (alpha)

    Reference:
      Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa

    Returns
    -------
    numpy.ndarray
        Thermal expansion coefficient in deg C -1
    """
    return -1.0 * (drho_dtemp(T, S, p) / density(T, S, p))


def beta(T, S, p):
    """Calculate haline contraction coefficient (beta)

    Reference:
      Wright, 1997, J. Atmos. Ocean. Tech., 14, 735-740.

    Parameters
    ----------
    T : numpy.ndarray
        Sea water potential temperature in deg C
    S : numpy.ndarray
        Sea water practical salinity in PSU
    p : numpy.ndarray
        Sea water absolute pressure in Pa

    Returns
    -------
    numpy.ndarray
        Haline contraction coefficient PSU -1
    """
    return drho_dsal(T, S, p) / density(T, S, p)
