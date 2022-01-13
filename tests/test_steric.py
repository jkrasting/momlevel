import xarray as xr
import numpy as np
from momlevel import steric, thermosteric, halosteric
from momlevel.eos import wright

np.random.seed(123)
time = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"))
z_l = xr.DataArray(np.array([2.5, 50.0, 100.0, 1000.0, 5000.0]), dims=("z_l"))
xh = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="xh")
yh = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="yh")

thetao = xr.DataArray(
    np.random.normal(15.0, 5.0, (5, 5, 5, 5)),
    dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
)
so = xr.DataArray(
    np.random.normal(35.0, 1.5, (5, 5, 5, 5)),
    dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
)
volcello = xr.DataArray(
    np.random.normal(1000.0, 100.0, (5, 5, 5, 5)),
    dims=({"time": time, "z_l": z_l, "yh": yh, "xh": xh}),
)
areacello = xr.DataArray(
    np.random.normal(100.0, 10.0, (5, 5)), dims=({"yh": yh, "xh": xh})
)
areacello = areacello / areacello.sum()
areacello = areacello * 3.6111092e14

dset = xr.Dataset(
    {"thetao": thetao, "so": so, "volcello": volcello, "areacello": areacello}
)
dset = dset.assign_coords({"time": time, "z_l": z_l, "yh": yh, "xh": xh})


def test_steric_broadcast():
    result = steric(dset)
    reference = float(result["reference_rho"][1, 2, 3])
    rho = wright(
        float(dset["thetao"][0, 1, 2, 3]),
        float(dset["so"][0, 1, 2, 3]),
        float(dset["z_l"][1]) * 1.0e4,
    )
    assert np.allclose(reference, rho)


reference_results = {
    "reference_thetao": 1892.9343653921171,
    "reference_so": 4386.6843782162,
    "reference_vol": 125401.8625239444,
    "reference_rho": 128880.60377451136,
    "reference_height": 8.74107451e-09,
    "global_reference_height": 4.340836e-08,
    "global_reference_vol": 125401.8625239444,
    "global_reference_rho": 1030.9696145532507,
}


def test_halosteric_values():
    result = halosteric(dset).sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["reference_height"]
    )
    assert np.allclose(result["expansion_coeff"], 0.079859382545713)
    assert np.allclose(result["halosteric"], 5.28555377e-12)


def test_steric_values():
    result = steric(dset).sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["reference_height"]
    )
    assert np.allclose(result["expansion_coeff"], 0.0321479462294750)
    assert np.allclose(result["steric"], 1.55378385e-12)


def test_thermosteric_values():
    result = thermosteric(dset).sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["reference_height"]
    )
    assert np.allclose(result["expansion_coeff"], -0.0485257018113030)
    assert np.allclose(result["thermosteric"], -3.79231742e-12)


def test_halosteric_global_values():
    result = halosteric(dset, domain="global").sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(
        result["global_reference_vol"], reference_results["global_reference_vol"]
    )
    assert np.allclose(
        result["global_reference_rho"], reference_results["global_reference_rho"]
    )
    assert np.allclose(result["expansion_coeff"], 0.07137665705082741)
    assert np.allclose(result["halosteric"], 1.98293992e-13)


def test_steric_global_values():
    result = steric(dset, domain="global").sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(
        result["global_reference_vol"], reference_results["global_reference_vol"]
    )
    assert np.allclose(
        result["global_reference_rho"], reference_results["global_reference_rho"]
    )
    assert np.allclose(result["expansion_coeff"], 0.022642849677982482)
    assert np.allclose(result["steric"], 6.29048941e-14)


def test_thermosteric_global_values():
    result = thermosteric(dset, domain="global").sum()
    assert np.allclose(
        result["reference_thetao"], reference_results["reference_thetao"]
    )
    assert np.allclose(result["reference_so"], reference_results["reference_so"])
    assert np.allclose(result["reference_vol"], reference_results["reference_vol"])
    assert np.allclose(result["reference_rho"], reference_results["reference_rho"])
    assert np.allclose(
        result["reference_height"], reference_results["global_reference_height"]
    )
    assert np.allclose(
        result["global_reference_vol"], reference_results["global_reference_vol"]
    )
    assert np.allclose(
        result["global_reference_rho"], reference_results["global_reference_rho"]
    )
    assert np.allclose(result["expansion_coeff"], -0.049692744362344)
    assert np.allclose(result["thermosteric"], -1.38053154e-13)
