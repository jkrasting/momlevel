import pytest

import hashlib
import numpy as np
import pandas as pd
import pkg_resources as pkgr

from momlevel import reference
from momlevel import util
from momlevel.test_data import (
    generate_test_data,
    generate_test_data_dz,
    generate_test_data_time,
    generate_test_data_uv,
)

from momlevel.test_data.time import generate_daily_timeaxis


def test_generate_daily_timeaxis():
    assert len(generate_daily_timeaxis()) == 730
    assert len(generate_daily_timeaxis(calendar="standard")) == 731


dset = generate_test_data()
dset2 = generate_test_data_dz()
dset3 = generate_test_data_time()
dset4 = generate_test_data_time(calendar="julian")
dset5 = generate_test_data_uv()

dset6 = generate_test_data_time(
    nyears=2, start_year=1979, frequency="D", calendar="noleap"
)

dset7 = generate_test_data_time(
    nyears=2, start_year=1979, frequency="D", calendar="standard"
)

dset8 = generate_test_data_time(
    nyears=5, start_year=1979, frequency="D", calendar="noleap"
)

dset9 = generate_test_data_time(
    nyears=5, start_year=1979, frequency="D", calendar="standard"
)


def test_default_coords_1():
    result = util.default_coords()
    assert result == ("time", "z_l", "z_i")


def test_default_coords_2():
    coord_names = {"z": "lev", "t": "TIME"}
    result = util.default_coords(coord_names=coord_names)
    assert result == ("TIME", "lev", "z_i")


def test_validate_areacello_1():
    assert util.validate_areacello(dset.areacello)


def test_validate_areacello_2():
    with pytest.raises(Exception):
        assert util.validate_areacello(dset.areacello * 1.3)


def test_validate_dataset_1():
    util.validate_dataset(dset)


def test_validate_dataset_2():
    """tests that missing variable raises an exception"""
    test_dset = dset.copy()
    test_dset = test_dset.drop_vars(["thetao"])
    with pytest.raises(Exception):
        util.validate_dataset(test_dset)


def test_validate_dataset_3():
    """tests that incorrect area raises an exception"""
    test_dset = dset.copy()
    test_dset["areacello"] = test_dset["areacello"] * 1.3
    with pytest.raises(Exception):
        util.validate_dataset(test_dset)


def test_validate_dataset_4():
    """tests that incorrect area issues a warning"""
    test_dset = dset.copy()
    test_dset["areacello"] = test_dset["areacello"] * 1.3
    with pytest.warns(UserWarning):
        util.validate_dataset(test_dset, strict=False)


def test_validate_dataset_5():
    """tests that ref dataset has no time dim"""
    test_dset = dset.copy()
    with pytest.raises(Exception):
        util.validate_dataset(test_dset, reference=True)


def test_validate_dataset_6():
    """tests that reference dataset is valid"""
    ref_dset = reference.setup_reference_state(dset, eos="Wright")
    util.validate_dataset(ref_dset, reference=True)


def test_validate_dataset_7():
    """tests that missing reference dataset field raises an exception"""
    ref_dset = reference.setup_reference_state(dset, eos="Wright")
    ref_dset = ref_dset.drop_vars(["rhoga"])
    with pytest.raises(Exception):
        util.validate_dataset(ref_dset, reference=True)


def test_validate_dataset_8():
    """tests additional variables option"""
    test_dset = dset.copy()
    additional_vars = ["foo", "bar"]
    with pytest.raises(Exception):
        util.validate_dataset(test_dset, additional_vars=additional_vars)


def test_annual_average_1():
    """tests annual average of a noleap calendar dataset"""
    result = util.annual_average(dset3).sum()
    assert np.allclose(result["var_a"], 12540.38661327)
    assert np.allclose(result["var_b"], 12513.3738587)


def test_annual_average_2():
    """tests annual average of a julian calendar dataset"""
    result = util.annual_average(dset4).sum()
    assert np.allclose(result["var_a"], 12540.37420516)
    assert np.allclose(result["var_b"], 12513.42390321)


def test_annual_average_3():
    """tests annual average of a noleap calendar data array"""
    result = util.annual_average(dset3["var_a"]).sum()
    assert np.allclose(result, 12540.38661327)


def test_annual_average_4():
    """tests annual average of a julian calendar dataset"""
    result = util.annual_average(dset4["var_a"]).sum()
    assert np.allclose(result, 12540.37420516)


def test_get_xgcm_grid_1():
    """tests xgcm grid construction for non-symmetric input"""
    result = util.get_xgcm_grid(dset5)
    answer = dict({"center": "right", "right": "center"})
    assert result.__dict__["axes"]["X"].__dict__["_default_shifts"] == answer
    assert result.__dict__["axes"]["Y"].__dict__["_default_shifts"] == answer


def test_get_xgcm_grid_2():
    """tests xgcm grid construction for symmetric input"""
    result = util.get_xgcm_grid(dset5, symmetric=True)
    answer = dict({"center": "outer", "outer": "center"})
    assert result.__dict__["axes"]["X"].__dict__["_default_shifts"] == answer
    assert result.__dict__["axes"]["Y"].__dict__["_default_shifts"] == answer


def test_validate_tidegauge_data_1():
    """tests that input datasets to the tide gauge routines are valid"""
    util.validate_tidegauge_data(dset.thetao, "xh", "yh", None)


def test_validate_tidegauge_data_2():
    """tests that input datasets to the tide gauge routines are valid"""
    with pytest.raises(Exception):
        util.validate_tidegauge_data(dset, "xh", "yh", None)


def test_validate_tidegauge_data_3():
    """tests that input datasets to the tide gauge routines are valid"""
    with pytest.raises(Exception):
        util.validate_tidegauge_data(dset.thetao, "geolon", "geolat", None)


def test_validate_tidegauge_data_4():
    """tests that input datasets to the tide gauge routines are valid"""
    util.validate_tidegauge_data(dset.thetao, dset.geolon, dset.geolat, None)


def test_validate_tidegauge_data_5():
    """tests that input datasets to the tide gauge routines are valid"""
    with pytest.raises(Exception):
        util.validate_tidegauge_data(
            dset.thetao, dset.geolon, np.array(dset.geolat), None
        )


def test_validate_tidegauge_data_6():
    """tests that input datasets to the tide gauge routines are valid"""
    util.validate_tidegauge_data(
        dset.thetao, dset.geolon, dset.geolat, dset.areacello * 0.0
    )


def test_validate_tidegauge_data_7():
    """tests that input datasets to the tide gauge routines are valid"""
    with pytest.raises(Exception):
        util.validate_tidegauge_data(dset.thetao, dset.geolon, dset.geolat, "wet")


def test_tile_nominal_coords():
    result1, result2 = util.tile_nominal_coords(dset.xh, dset.yh)
    assert result1.sum().values == result2.sum().values
    assert np.allclose(result1.sum(), 75.0)


def test_geolocate_points():
    """Tests behavior of geolocate_points function"""
    df_model = pd.read_csv(
        pkgr.resource_filename("momlevel", "resources/NWA12_grid_dataframe.csv"),
        index_col=[0, 1],
    )
    df_loc = pd.read_csv(
        pkgr.resource_filename("momlevel", "resources/us_tide_gauges.csv")
    )
    df_loc = df_loc.rename(columns={"PSMSL_site": "name"})
    reference = pd.read_csv(
        pkgr.resource_filename("momlevel", "resources/geolocate_points_reference.csv"),
        index_col=[0],
    )
    results = util.geolocate_points(df_model, df_loc, threshold=13.75)
    assert np.allclose(results["distance"], reference["distance"], rtol=1e-04)


def test_get_pv_colormap():
    levels, colors = util.get_pv_colormap()
    m = hashlib.md5()
    for s in levels + colors:
        m.update(str(s).encode())
    assert m.hexdigest() == "29b7e26115ca782ffa09994057137f1a"


def test_monthly_average_1():
    result = util.monthly_average(dset6).sum()
    assert np.allclose(result["var_a"], 60105.04603946)
    assert np.allclose(result["var_b"], 59859.46422535)


def test_monthly_average_2():
    result = util.monthly_average(dset7).sum()
    assert np.allclose(result["var_a"], 60110.203595)
    assert np.allclose(result["var_b"], 59858.37293512)


def test_annual_cycle_1():
    result = util.annual_cycle(util.monthly_average(dset8))
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 30015.57996061)
    assert np.allclose(result["var_b"], 29961.89265959)


def test_annual_cycle_2():
    result = util.annual_cycle(util.monthly_average(dset9))
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 30015.59638431)
    assert np.allclose(result["var_b"], 29961.53401375)


def test_annual_cycle_3():
    result = util.annual_cycle(util.monthly_average(dset8), func="std")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 890.24286612)
    assert np.allclose(result["var_b"], 917.12436607)


def test_annual_cycle_4():
    result = util.annual_cycle(util.monthly_average(dset8), func="max")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 31248.84868587)
    assert np.allclose(result["var_b"], 31237.81311579)


def test_annual_cycle_5():
    result = util.annual_cycle(util.monthly_average(dset8), func="min")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 28788.98557133)
    assert np.allclose(result["var_b"], 28705.85687133)
