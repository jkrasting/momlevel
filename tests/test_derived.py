import pytest
import numpy as np
import xarray as xr

from momlevel import derived
from momlevel.test_data import generate_test_data
from momlevel.test_data import generate_test_data_dz
from momlevel.test_data import generate_test_data_uv

dset1 = generate_test_data()
dset2 = generate_test_data_dz()
dset3 = generate_test_data_uv()


def test_adjust_negative_n2():
    obvfsq = derived.calc_n2(dset1.thetao, dset1.so)
    adjusted = derived.adjust_negative_n2(obvfsq)
    assert np.allclose(adjusted.sum(), 0.12093286)


def test_calc_coriolis():
    coriolis = derived.calc_coriolis(dset1.geolat)
    assert np.allclose(coriolis.sum(), 2.71050543e-20)


def test_calc_dz_1():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho)
    assert np.allclose(dz.sum(), 1130.67307641)


def test_calc_dz_2():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho, fraction=True)
    assert np.allclose(dz.sum(), 85.53726628)


def test_calc_dz_3():
    deptho = dset2.deptho.copy()
    deptho[4, 4] = -200.0
    with pytest.raises(Exception):
        derived.calc_dz(dset2.z_l, dset2.z_i, deptho)


def test_calc_dz_4():
    dz = derived.calc_dz(dset2.z_l, dset2.z_i, dset2.deptho, top=12., bottom=33.)
    assert np.allclose(dz.sum(), 363.71725794)


def test_calc_rho():
    rho = derived.calc_rho(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    pytest.rho = rho
    assert np.allclose(rho.sum(), 643872.59725673)


def test_calc_n2_1():
    obvfsq = derived.calc_n2(dset1.thetao, dset1.so)
    assert np.allclose(obvfsq.sum(), 0.00338354)


def test_calc_n2_2():
    obvfsq = derived.calc_n2(dset1.thetao, dset1.so, adjust_negative=True)
    assert np.allclose(obvfsq.sum(), 0.12093286)


def test_calc_pdens_1():
    rhopot = derived.calc_pdens(dset1.thetao, dset1.so, eos="Wright")
    assert np.allclose(rhopot.sum(), 641182.68524632)


def test_calc_pdens_2():
    rhopot = derived.calc_pdens(dset1.thetao, dset1.so, level=2000.0, eos="Wright")
    assert np.allclose(rhopot.sum(), 646573.41064627)


def test_calc_alpha():
    alpha = derived.calc_alpha(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    assert np.allclose(alpha.sum(), 0.14302587)


def test_calc_beta():
    beta = derived.calc_beta(dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright")
    assert np.allclose(beta.sum(), 0.4639801)


def test_calc_masso():
    masso = derived.calc_masso(pytest.rho, dset1.volcello)
    pytest.masso = masso
    assert np.allclose(masso.sum(), 6.45215577e08)


def test_calc_volo_1():
    with pytest.raises(Exception):
        _ = derived.calc_volo(dset1.volcello)


def test_calc_volo_2():
    volo = derived.calc_volo(dset1.volcello.isel(time=0))
    pytest.volo = volo
    assert np.allclose(volo, 125921.15458782)


def test_rhoga():
    rhoga = derived.calc_rhoga(pytest.masso, pytest.volo)
    assert np.allclose(rhoga.sum(), 5123.96490958)


def test_calc_rel_vort():
    result = derived.calc_rel_vort(dset3)
    assert np.allclose(result.sum(), -6.92989256e-14)


def test_calc_pv_1():
    zeta = derived.calc_rel_vort(dset3)
    n2 = derived.calc_n2(dset1.thetao, dset1.so)
    pv = derived.calc_pv(zeta, dset3.Coriolis, n2, units="m")
    assert np.allclose(pv.sum(), -7.97291438e-08)


def test_calc_pv_2():
    zeta = derived.calc_rel_vort(dset3)
    n2 = derived.calc_n2(dset1.thetao, dset1.so)
    pv = derived.calc_pv(zeta, dset3.Coriolis, n2, units="cm")
    assert np.allclose(pv.sum(), 584073.75980102)


def test_calc_rossby_rd():
    n2 = derived.calc_n2(dset1.thetao, dset1.so)
    dz = derived.calc_dz(dset1.z_l, dset1.z_i, dset1.deptho)
    wave_speed = derived.calc_wave_speed(n2, dz)
    coriolis = derived.calc_coriolis(dset1.geolat)
    rossby_rd = derived.calc_rossby_rd(wave_speed, coriolis)
    rossby_rd = xr.where(np.isinf(rossby_rd), np.nan, rossby_rd)
    assert np.allclose(rossby_rd.sum(), 4443140.80206)


def test_calc_spice():
    pi = derived.calc_spice(dset1.thetao, dset1.so)
    assert np.allclose(pi.sum(), 1412.03593361)


def test_calc_stability_angle():
    tu_ang = derived.calc_stability_angle(
        dset1.thetao, dset1.so, dset1.z_l * 1.0e4, eos="Wright"
    )
    assert np.allclose(tu_ang.sum(), 5838.68533435)


def test_calc_wave_speed():
    n2 = derived.calc_n2(dset1.thetao, dset1.so)
    dz = derived.calc_dz(dset1.z_l, dset1.z_i, dset1.deptho)
    wave_speed = derived.calc_wave_speed(n2, dz)
    assert np.allclose(wave_speed.sum(), 524.30956095)
