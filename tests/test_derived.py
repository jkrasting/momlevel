import pytest
import numpy as np

from momlevel import derived
from momlevel.eos import wright
from momlevel.test_data import generate_test_data

dset = generate_test_data()


def test_calc_rho():
    rho = derived.calc_rho(wright, dset.thetao, dset.so, dset.z_l * 1.0e4)
    pytest.rho = rho
    assert np.allclose(rho.sum(), 644369.50302943)


def test_calc_masso():
    masso = derived.calc_masso(pytest.rho, dset.volcello)
    pytest.masso = masso
    assert np.allclose(masso.sum(), 6.43587532e08)


def test_calc_volo_1():
    with pytest.raises(Exception):
        result = derived.calc_volo(dset.volcello)


def test_calc_volo_1():
    volo = derived.calc_volo(dset.volcello.isel(time=0))
    pytest.volo = volo
    assert np.allclose(volo, 125401.86252394)


def test_rhoga():
    rhoga = derived.calc_rhoga(pytest.masso, pytest.volo)
    assert np.allclose(rhoga.sum(), 5132.20074272)
