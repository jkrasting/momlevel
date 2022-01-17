import pytest
import numpy as np

from momlevel import derived
from momlevel.test_data import generate_test_data
from momlevel.test_data import generate_test_data_dz

dset1 = generate_test_data()
dset2 = generate_test_data_dz()


def test_calc_dz_1():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho)
    assert np.allclose(dz.sum(), 1261.33383326)


def test_calc_dz_2():
    deptho = dset2.deptho.copy()
    deptho[4, 4] = -200.0
    with pytest.raises(Exception):
        derived.calc_dz(dset2.z_l, dset2.z_i, deptho)


def test_calc_rho():
    rho = derived.calc_rho(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    pytest.rho = rho
    assert np.allclose(rho.sum(), 644369.50302943)


def test_calc_masso():
    masso = derived.calc_masso(pytest.rho, dset1.volcello)
    pytest.masso = masso
    assert np.allclose(masso.sum(), 6.43587532e08)


def test_calc_volo_1():
    with pytest.raises(Exception):
        result = derived.calc_volo(dset1.volcello)


def test_calc_volo_1():
    volo = derived.calc_volo(dset1.volcello.isel(time=0))
    pytest.volo = volo
    assert np.allclose(volo, 125401.86252394)


def test_rhoga():
    rhoga = derived.calc_rhoga(pytest.masso, pytest.volo)
    assert np.allclose(rhoga.sum(), 5132.20074272)
