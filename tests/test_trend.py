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


def test_time_conversion_factor():
    assert trend.time_conversion_factor("ns", "ns") == 1.0
    assert trend.time_conversion_factor("yr", "day") == 365.0
    assert trend.time_conversion_factor("day", "hr") == 24.0
    assert trend.time_conversion_factor("day", "s") == 86400.0
    assert np.allclose(trend.time_conversion_factor("mon", "day"), 30.417)


def test_calc_linear_trend_1():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    result = trend.calc_linear_trend(dset_in.var_a)
    assert np.allclose(result.var_a_slope.sum(), -2.16505389e-17)
    assert np.allclose(result.var_a_intercept.sum(), 2511.47295749)
    assert result.var_a_slope.units == " ns-1"


def test_calc_linear_trend_2():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    result = trend.calc_linear_trend(dset_in.var_a, time_units="yr")
    assert np.allclose(result.var_a_slope.sum(), -0.68277139)
    assert np.allclose(result.var_a_intercept.sum(), 2511.47295749)
    assert result.var_a_slope.units == " yr-1"


def test_broadcast_trend_1():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    slope = trend.calc_linear_trend(dset_in.var_a)
    result = trend.broadcast_trend(slope.var_a_slope,dset_in.time)
    assert np.allclose(result.sum(),-14329.66462312)


def test_broadcast_trend_2():
    dset_in = dset8.drop_vars(["time_bnds", "average_T1", "average_T2", "average_DT"])
    slope = trend.calc_linear_trend(dset_in.var_a, time_units="yr")
    result = trend.broadcast_trend(slope.var_a_slope,dset_in.time)
    assert np.allclose(result.sum(),-14329.66462312)