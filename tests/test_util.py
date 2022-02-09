import pytest

import numpy as np

from momlevel import reference
from momlevel import util
from momlevel.test_data import (
    generate_test_data,
    generate_test_data_dz,
    generate_test_data_time,
    generate_test_data_uv,
)

dset = generate_test_data()
dset2 = generate_test_data_dz()
dset3 = generate_test_data_time()
dset4 = generate_test_data_time(calendar="julian")
dset5 = generate_test_data_uv()


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
