import xarray as xr
import numpy as np
from .thermosteric import thermosteric
from .eos.wright import wright

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

dset = xr.Dataset(
    {"thetao": thetao, "so": so, "volcello": volcello, "areacello": areacello}
)
dset = dset.assign_coords({"time": time, "z_l": z_l, "yh": yh, "xh": xh})


def test_thermosteric_broadcast():
    result = thermosteric(dset)
    reference = float(result["reference_rho"][1, 2, 3])
    rho = wright(
        float(dset["thetao"][0, 1, 2, 3]),
        float(dset["so"][0, 1, 2, 3]),
        float(dset["z_l"][1]) * 1.0e4,
    )
    assert np.allclose(reference, rho)


def test_thermosteric_values():
    result = thermosteric(dset).sum()
    assert np.allclose(result["reference_so"], 4386.684378216)
    assert np.allclose(result["reference_vol"], 125401.862523944)
    assert np.allclose(result["reference_rho"], 128880.6037745114)
    assert np.allclose(result["reference_height"], 1230.16465627079)
    assert np.allclose(result["expansion_coeff"], -0.0485257018113030)
    assert np.allclose(result["thermosteric"], -0.533707251686609)
