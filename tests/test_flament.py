import numpy as np
from momlevel.spice.flament import spice

S = np.arange(33.0, 37.1, 0.1)
T = np.arange(0.0, 31.0, 1.0)

SS = np.tile(S[None, :], (len(T), 1))
TT = np.tile(T[:, None], (1, len(S)))


def test_flament_spice():
    pi = spice(TT, SS)
    assert np.allclose(pi.sum(), 3283.680384169385)
