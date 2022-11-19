import numpy as np

from momlevel import util
from momlevel import trend
from momlevel.test_data import (
    generate_test_data_time,
)

dset8 = generate_test_data_time(
    nyears=5, start_year=1979, frequency="D", calendar="noleap"
)


def test_linear_detrend_1():
    result = util.linear_detrend(dset8.var_a[:, 0, 0])
    assert np.allclose(result.sum(), -6.05950845e-11)


def test_linear_detrend_2():
    result = trend.linear_detrend(dset8.var_a[:, 0, 0], mode="correct")
    assert np.allclose(result.sum(), 187044.59558497)


def test_linear_detrend_3():
    result = trend.linear_detrend(dset8.var_a)
    assert np.allclose(result.sum(), -1.26760824e-09)


def test_linear_detrend_4():
    result = trend.linear_detrend(dset8.var_a, mode="correct")
    assert np.allclose(result.sum(), 4583438.14742081)


def test_linear_detrend_5():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    result = trend.linear_detrend(dset_in, mode="correct")
    assert np.allclose(result.var_a.sum(), 4583438.14742081)
    assert np.allclose(result.var_b.sum(), 4589810.41140568)
