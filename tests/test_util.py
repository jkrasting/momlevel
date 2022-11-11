import pytest

import hashlib
import numpy as np
import pandas as pd
import pkg_resources as pkgr

from momlevel import reference
from momlevel import util
from momlevel.test_data import (
    generate_daily_timeaxis,
    generate_test_data,
    generate_test_data_dz,
    generate_test_data_time,
    generate_test_data_uv,
)



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


def test_validate_areacello_3():
    with pytest.raises(Exception):
        assert util.validate_areacello(dset.areacello, tolerance=1.0e-30)


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
    assert np.allclose(result["var_a"], 12484.37032342)
    assert np.allclose(result["var_b"], 12605.66490932)


def test_annual_average_2():
    """tests annual average of a julian calendar dataset"""
    result = util.annual_average(dset4).sum()
    assert np.allclose(result["var_a"], 12484.17097863)
    assert np.allclose(result["var_b"], 12605.18695941)


def test_annual_average_3():
    """tests annual average of a noleap calendar data array"""
    result = util.annual_average(dset3["var_a"]).sum()
    assert np.allclose(result, 12484.37032342)


def test_annual_average_4():
    """tests annual average of a julian calendar dataset"""
    result = util.annual_average(dset4["var_a"]).sum()
    assert np.allclose(result, 12484.17097863)


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
    assert np.allclose(result["var_a"], 60167.13927143)
    assert np.allclose(result["var_b"], 60036.90907922)


def test_monthly_average_2():
    result = util.monthly_average(dset7).sum()
    assert np.allclose(result["var_a"], 60163.23828842)
    assert np.allclose(result["var_b"], 60036.8317591)


def test_annual_cycle_1():
    result = util.annual_cycle(util.monthly_average(dset8))
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 30043.9981433)
    assert np.allclose(result["var_b"], 29992.27048348)


def test_annual_cycle_2():
    result = util.annual_cycle(util.monthly_average(dset9))
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 30043.89217891)
    assert np.allclose(result["var_b"], 29993.37508877)


def test_annual_cycle_3():
    result = util.annual_cycle(util.monthly_average(dset8), func="std")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 909.30443538)
    assert np.allclose(result["var_b"], 913.21989648)


def test_annual_cycle_4():
    result = util.annual_cycle(util.monthly_average(dset8), func="max")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 31295.58398947)
    assert np.allclose(result["var_b"], 31260.98038945)


def test_annual_cycle_5():
    result = util.annual_cycle(util.monthly_average(dset8), func="min")
    assert len(result.time) == 12
    result = result.sum()
    assert np.allclose(result["var_a"], 28782.31507282)
    assert np.allclose(result["var_b"], 28742.60226595)


def test_linear_detrend_1():
    result = util.linear_detrend(dset8.var_a[:, 0, 0])
    assert np.allclose(result.sum(), -6.05950845e-11)


def test_linear_detrend_2():
    result = util.linear_detrend(dset8.var_a[:, 0, 0], mode="correct")
    assert np.allclose(result.sum(), 187044.59558497)


def test_linear_detrend_3():
    result = util.linear_detrend(dset8.var_a)
    assert np.allclose(result.sum(), -1.26760824e-09)


def test_linear_detrend_4():
    result = util.linear_detrend(dset8.var_a, mode="correct")
    assert np.allclose(result.sum(), 4583438.14742081)


def test_linear_detrend_5():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    result = util.linear_detrend(dset_in, mode="correct")
    assert np.allclose(result.var_a.sum(), 4583438.14742081)
    assert np.allclose(result.var_b.sum(), 4589810.41140568)
