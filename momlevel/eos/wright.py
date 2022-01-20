""" wright.py -- Wright Equation of State """

__all__ = ["density"]

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
        Sea water pressure in Pa

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
