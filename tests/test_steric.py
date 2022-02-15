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
    "reference_thetao": 1892.9343653921171,
    "reference_so": 4386.6843782162,
    "reference_vol": 125401.8625239444,
    "reference_rho": 128781.84605852,
    "reference_height": 8.74107451e-09,
    "global_reference_height": 3.4726688e-10,
    "global_reference_vol": 125401.8625239444,
    "global_reference_rho": 1030.19308179,
}


def test_halosteric_values():
    result, reference = halosteric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["halosteric"], 27.92630768)
    assert np.allclose(result["delta_rho"], -83.05163718)


def test_steric_values():
    result, reference = steric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["steric"], 17.99510466)
    assert np.allclose(result["delta_rho"], -34.87805193)


def test_thermosteric_values():
    result, reference = thermosteric(dset)
    result = result.sum()
    reference = reference.sum()
    assert np.allclose(reference["thetao"], reference_results["reference_thetao"])
    assert np.allclose(reference["so"], reference_results["reference_so"])
    assert np.allclose(reference["volcello"], reference_results["reference_vol"])
    assert np.allclose(reference["rho"], reference_results["reference_rho"])
    assert np.allclose(result["thermosteric"], -10.0255142)
    assert np.allclose(result["delta_rho"], 49.15951991)


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
    assert np.allclose(reference["thetao"], 1957.37033788)
    assert np.allclose(reference["so"], 4382.2088354)
    assert np.allclose(reference["volcello"], 125023.36640225)
    assert np.allclose(reference["rho"], 128769.07712858)
    assert np.allclose(result["steric"], -9.48781806)


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
    assert np.allclose(result["steric"], 13.93175417)
    assert np.allclose(result["delta_rho"], -18.10758127)
