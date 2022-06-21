import numpy as np

from momlevel import inverse_barometer
from momlevel.test_data import generate_test_data

dset = generate_test_data().isel(z_l=0)

def test_inverse_barometer():
    result = inverse_barometer(dset.thetao,dset.so,101325.)
    assert np.allclose(result.sum(),-1260.12620941)
