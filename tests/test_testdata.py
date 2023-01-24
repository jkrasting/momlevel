import dask

dask.config.set({"tokenize.ensure-deterministic": True})

import momlevel.test_data as test_data


def test_generate_daily_timeaxis_1():
    result = test_data.time.generate_daily_timeaxis()
    assert len(result) == 730


def test_generate_daily_timeaxis_2():
    result = test_data.time.generate_daily_timeaxis(calendar="standard")
    assert len(result) == 731


def test_generate_time_stub_1():
    result = test_data.generate_time_stub()
    assert len(result.time) == 60


def test_generate_time_stub_2():
    result = test_data.generate_time_stub(frequency="D")
    assert len(result.time) == 1825


def test_generate_time_stub_3():
    result = test_data.generate_time_stub()
    result = (
        result.time_bnds.values[0][0].isoformat(),
        result.time_bnds.values[0][1].isoformat(),
    )
    assert result == ("1981-01-01T00:00:00", "1981-02-01T00:00:00")


def test_xyfields_1():
    result = test_data.tripolar.horizontal.xy_fields()
    assert dask.base.tokenize(result) == "c28eb260bcb023cd6f78b0edf93d0ec1"


def test_xyfields_2():
    result = test_data.tripolar.horizontal.xy_fields(seed=999)
    assert dask.base.tokenize(result) == "493fc761a28c3ddfc9f099d51ffbf3d1"


def test_xyfields_3():
    result = test_data.tripolar.horizontal.xy_fields(point="u")
    assert dask.base.tokenize(result) == "5edb2afe09a24b2cea9f1fc3dd44ffdc"


def test_xyfields_4():
    result = test_data.tripolar.horizontal.xy_fields(point="v")
    assert dask.base.tokenize(result) == "4cef08a13817d9d709b9bd56a7eb4d6c"


def test_xyfields_5():
    result = test_data.tripolar.horizontal.xy_fields(point="c")
    assert dask.base.tokenize(result) == "5dc886d24e974816a885e6233b161564"


def test_zlevel_fields():
    result = test_data.tripolar.vertical.zlevel_fields(include_deptho=False)
    assert dask.base.tokenize(result) == "c7e26f7ba195dd02a154bb561a0f00e7"


def test_zlevel_fields_2():
    result = test_data.tripolar.vertical.zlevel_fields()
    assert dask.base.tokenize(result) == "419f6955760d100b1da430f4c662d5ba"


def test_generate_test_data():
    result = test_data.generate_test_data()
    assert dask.base.tokenize(result) == "3a03152e9c749e616e4e8ffdd8eaa417"


def test_generate_test_data_dz():
    result = test_data.generate_test_data_dz()
    assert dask.base.tokenize(result) == "0aa0d8dbeb0705634252441998499ec2"


def test_generate_test_data_uv():
    result = test_data.generate_test_data_uv()
    assert dask.base.tokenize(result) == "641cdd0109b2b00fdcbe2a16b8edf70d"
