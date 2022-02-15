import pytest
import numpy as np

from momlevel import derived
from momlevel.test_data import generate_test_data
from momlevel.test_data import generate_test_data_dz
from momlevel.test_data import generate_test_data_uv

dset1 = generate_test_data()
dset2 = generate_test_data_dz()
dset3 = generate_test_data_uv()


def test_calc_dz_1():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho)
    assert np.allclose(dz.sum(), 1261.33383326)


def test_calc_dz_2():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho, fraction=True)
    assert np.allclose(dz.sum(), 91.82404981)


def test_calc_dz_3():
    deptho = dset2.deptho.copy()
    deptho[4, 4] = -200.0
    with pytest.raises(Exception):
        derived.calc_dz(dset2.z_l, dset2.z_i, deptho)


def test_calc_rho():
    rho = derived.calc_rho(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    pytest.rho = rho
    assert np.allclose(rho.sum(), 643847.01494266)


def test_calc_n2():
    obvfsq = derived.calc_n2(dset1.thetao, dset1.so)
    assert np.allclose(obvfsq.sum(), 0.11750034)


def test_calc_pdens_1():
    rhopot = derived.calc_pdens(dset1.thetao, dset1.so, eos="Wright")
    assert np.allclose(rhopot.sum(), 641153.07032298)


def test_calc_pdens_2():
    rhopot = derived.calc_pdens(dset1.thetao, dset1.so, level=2000.0, eos="Wright")
    assert np.allclose(rhopot.sum(), 646547.38808142)


def test_calc_alpha():
    alpha = derived.calc_alpha(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    assert np.allclose(alpha.sum(), 0.14270076)


def test_calc_beta():
    beta = derived.calc_beta(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    assert np.allclose(beta.sum(), 0.46398704)


def test_calc_masso():
    masso = derived.calc_masso(pytest.rho, dset1.volcello)
    pytest.masso = masso
    assert np.allclose(masso.sum(), 6.43066545e08)


def test_calc_volo_1():
    with pytest.raises(Exception):
        result = derived.calc_volo(dset1.volcello)


def test_calc_volo_1():
    volo = derived.calc_volo(dset1.volcello.isel(time=0))
    pytest.volo = volo
    assert np.allclose(volo, 125401.86252394)


def test_rhoga():
    rhoga = derived.calc_rhoga(pytest.masso, pytest.volo)
    assert np.allclose(rhoga.sum(), 5128.04620652)


def test_calc_rel_vort():
    result = derived.calc_rel_vort(dset3)
    assert np.allclose(result.sum(), -6.92989256e-14)


def test_calc_pv():
    zeta = derived.calc_rel_vort(dset3)
    n2 = derived.calc_n2(dset1.thetao, dset1.so)
    pv = derived.calc_pv(zeta, dset3.Coriolis, n2)
    # convert to WOCE conventional units of 10*14 cm-1 s-1
    pv = (pv / 100.0) * 1e14
    assert np.allclose(pv.sum(), 119787.96470602)
