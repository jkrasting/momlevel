import pytest
import numpy as np

from momlevel import steric, thermosteric, halosteric
from momlevel.eos import wright
from momlevel.test_data import generate_test_data

dset = generate_test_data()
dset2 = generate_test_data(seed=999)
dset3 = generate_test_data(start_year=1983, nyears=2, calendar="julian")


def test_steric_broadcast():
    result, reference = steric(dset)
    reference = float(reference["rho"][1, 2, 3])
    patm = 101325.0
    rho = wright.density(
        float(dset["thetao"][0, 1, 2, 3]),
        float(dset["so"][0, 1, 2, 3]),
        (float(dset["z_l"][1]) * 1.0e4) + patm,
    )
    assert np.allclose(reference, rho)


def test_steric_incorrect_area():
    _dset = dset.copy()
    _dset["areacello"] = _dset["areacello"] * 1.3
    with pytest.raises(Exception):
        _ = steric(_dset)


reference_results = {
    "reference_thetao": 1921.05772939,
    "reference_so": 4388.81731882,
    "reference_vol": 125921.15458782,
    "reference_rho": 128781.63975736,
    "reference_height": 8.74107451e-09,
    "global_reference_height": 3.4726688e-10,
    "global_reference_vol": 125921.15458782,
    "global_reference_rho": 1030.2309221,
}


def test_halosteric_values():
    result, reference = halosteric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["halosteric"], 4.39398075)
    assert np.allclose(result["delta_rho"], -32.07946717)


def test_steric_values():
    result, reference = steric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["steric"], 1.38250197)
    assert np.allclose(result["delta_rho"], -11.33133173)


def test_thermosteric_values():
    result, reference = thermosteric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["thermosteric"], -4.14327109)
    assert np.allclose(result["delta_rho"], 33.83631611)


def test_halosteric_global_values():
    result, reference = halosteric(dset, domain="global")
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(reference["volo"], reference_results["global_reference_vol"])
    assert np.allclose(reference["rhoga"], reference_results["global_reference_rho"])
    assert np.allclose(result["halosteric"], 1.98293992e-13)


def test_steric_global_values():
    result, reference = steric(dset, domain="global")
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(reference["volo"], reference_results["global_reference_vol"])
    assert np.allclose(reference["rhoga"], reference_results["global_reference_rho"])
    assert np.allclose(result["steric"], 6.29048941e-14)


def test_thermosteric_global_values():
    result, reference = thermosteric(dset, domain="global")
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(reference["volo"], reference_results["global_reference_vol"])
    assert np.allclose(reference["rhoga"], reference_results["global_reference_rho"])
    assert np.allclose(result["thermosteric"], -1.38053154e-13)


def test_steric_read_reference():
    _, reference = steric(dset2)
    result, reference = steric(dset, verbose=True, reference=reference)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], 1917.31113456)
    assert np.allclose(reference["so"], 4387.69334037)
    assert np.allclose(reference["volcello"], 125846.22269117)
    assert np.allclose(reference["rho"], 128780.12974804)
    assert np.allclose(result["steric"], 1.25554742)


def test_encoding_1():
    result, reference = steric(dset)
    assert result["delta_rho"].encoding["dtype"] == "float32"
    assert result["steric"].encoding["dtype"] == "float32"
    result, reference = steric(dset, dtype="float64")
    assert result["delta_rho"].encoding["dtype"] == "float64"
    assert result["steric"].encoding["dtype"] == "float64"


def test_encoding_2():
    result, reference = steric(dset, domain="global")
    assert result["reference_height"].encoding["dtype"] == "float32"
    assert result["steric"].encoding["dtype"] == "float32"
    result, reference = steric(dset, domain="global", dtype="float64")
    assert result["reference_height"].encoding["dtype"] == "float64"
    assert result["steric"].encoding["dtype"] == "float64"


def test_steric_annual_average():
    result, reference = steric(dset3, annual=True)
    assert len(result["time"]) == 2
    result = result.sum()
    assert np.allclose(result["steric"], 1.07892738)
    assert np.allclose(result["delta_rho"], -4.15906613)
