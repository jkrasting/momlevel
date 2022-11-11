""" flament.py -- P. Flament (2002) Spiciness calculation """

import numpy as np

__all__ = ["spice"]

B_IJ = np.array(
    [
        0.0,
        7.7442e-1,
        -5.85e-3,
        -9.84e-4,
        -2.06e-4,
        5.1655e-2,
        2.034e-3,
        -2.742e-4,
        -8.5e-6,
        1.36e-5,
        6.64783e-3,
        -2.4681e-4,
        -1.428e-5,
        3.337e-5,
        7.894e-6,
        -5.4023e-5,
        7.326e-6,
        7.0036e-6,
        -3.0412e-6,
        -1.0853e-6,
        3.949e-7,
        -3.029e-8,
        -3.8209e-7,
        1.0012e-7,
        4.7133e-8,
        -6.36e-10,
        -1.309e-9,
        6.048e-9,
        -1.1409e-9,
        -6.676e-10,
    ]
).reshape(6, 5)


def spice(thetao, so):
    """Function to calculate seawater spiciness

    This function uses the Flament 2002 methodology for calculating
    seawater spiciness:

    Flament, P. 2002: A state variable for characterizing water
    masses and their diffusive stability: spiciness.
    Progress in Oceanography, 54, 493–501.
    https://doi.org/10.1016/S0079-6611(02)00065-4

    Parameters
    ----------
    thetao : numpy.ndarray
        Sea water potential temperature in deg C
    so : numpy.ndarray
        Sea water practical salinity in PSU

    Returns
    -------
    numpy.ndarray
        Sea water spiciness, unitless
    """

    # convert non-numpy arguments to numpy arrays if necessary
    thetao = np.array([float(thetao)]) if isinstance(thetao, (float, int)) else thetao

    so = np.array([float(so)]) if isinstance(so, (float, int)) else so

    thetao_shape = thetao.shape
    so_shape = so.shape

    assert thetao_shape == so_shape, "thetao and so must have the same shape"

    # flatten multi-dimensional arrays for speed
    thetao = thetao.flatten()
    so = so.flatten()

    # expand terms
    thetao = np.swapaxes(np.array([thetao**x for x in range(0, 6)]), 0, 1)
    so = np.swapaxes(np.array([(so - 35.0) ** x for x in range(0, 5)]), 0, 1)

    # calculate spice
    pi = (
        np.tile(np.array(so)[:, None, :], (1, 6, 1))
        * np.tile(np.array(thetao)[:, :, None], (1, 1, 5))
        * B_IJ
    ).sum(axis=(-2, -1))

    # return array to original shape
    pi = pi.reshape(thetao_shape)

    return pi
