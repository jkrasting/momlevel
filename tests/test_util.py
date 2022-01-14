import pytest
import numpy as np

from momlevel import eos
from momlevel import reference
from momlevel import util
from momlevel.test_data import generate_test_data

dset = generate_test_data()


def test_default_coords_1():
    result = util.default_coords()
    assert result == ("time", "z_l")


def test_default_coords_2():
    coord_names = {"z": "lev", "t": "TIME"}
    result = util.default_coords(coord_names=coord_names)
    assert result == ("TIME", "lev")


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
    ref_dset = reference.setup_reference_state(dset, eos.wright)
    util.validate_dataset(ref_dset, reference=True)


def test_validate_dataset_7():
    """tests that missing reference dataset field raises an exception"""
    ref_dset = reference.setup_reference_state(dset, eos.wright)
    ref_dset = ref_dset.drop_vars(["rhoga"])
    with pytest.raises(Exception):
        util.validate_dataset(ref_dset, reference=True)
