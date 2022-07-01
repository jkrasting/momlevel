"""test_tidegauge.py - unit tests for tidegauge module"""

import pkg_resources as pkgr
import xarray as xr
import numpy as np
from momlevel import tidegauge

ds_nwa = xr.open_dataset(
    pkgr.resource_filename("momlevel", "resources/NWA12_sample_grid_data.nc")
)


def test_extract_tidegauge_1():
    """Tests tide gauge site extraction"""
    result = tidegauge.extract_tidegauge(
        ds_nwa.ssh_max,
        xcoord=ds_nwa.geolon,
        ycoord=ds_nwa.geolat,
        mask=ds_nwa.wet,
        threshold=13.75,
    )

    assert np.allclose(result["ATLANTIC_CITY"].sum(), 7.78345)
    assert np.allclose(result["BRIDGEPORT"].sum(), 9.865859)
    assert np.allclose(result["CAPE_MAY"].sum(), 7.3625193)
    assert np.allclose(result["DUCK_PIER_OUTSIDE"].sum(), 4.141247)
    assert np.allclose(result["KIPTOPEKE_BEACH"].sum(), 2.6744587)
    assert np.allclose(result["LEWES"].sum(), 3.915421)
    assert np.allclose(result["MONTAUK"].sum(), 1.1313734)
    assert np.allclose(result["NANTUCKET_ISLAND"].sum(), -3.6923892)
    assert np.allclose(result["NEWPORT"].sum(), 5.7311196)
    assert np.allclose(result["OCEAN_CITY_INLET"].sum(), 4.6226077)
    assert np.allclose(result["OREGON_INLET_MARINA"].sum(), 3.198695)
    assert np.allclose(result["PORTLAND"].sum(), 28.47948)
    assert np.allclose(result["SANDY_HOOK"].sum(), 11.59208)
    assert np.allclose(result["SEAVEY_ISLAND"].sum(), 27.770094)
    assert np.allclose(result["SEWELLS_POINT"].sum(), -1.0597064)
    assert np.allclose(result["SOLOMONS_ISLAND"].sum(), -9.02204)
