""" test_data.py - module for generating test data """

import xarray as xr
import numpy as np

__all__ = [
    "generate_test_data",
    "generate_test_data_dz",
    "generate_test_data_time",
    "generate_test_data_uv",
]


def generate_test_data(start_year=1981, nyears=0, calendar="noleap", seed=123):
    """Function to generate dataset for unit testing

    This function generates a test dataset. It includes 5 points in the
    vertical, latitudinal, and longitudinal dimensions. The dataset
    contains random values for `thetao`, `so`, `volcello`, `deptho` and
    `areacello`.

    If nyears == 0 (default), the time coordinate is a 5 point integer
    array. If nyears >= 1, a real world monthly time axis is generated.

    Parameters
    ----------
    start_year : int, optional
        Starting year, by default 1981
    nyears : int, optional
        Number of years to generate, by default 0
    calendar : str, optional
        CF-time recognized calendar, by default "noleap"
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        ntimes x 5 x 5 x 5 point dataset for unit testing
    """

    if nyears >= 1:
        dset = generate_time_stub(
            start_year=start_year, nyears=nyears, calendar=calendar
        )
    else:
        dset = xr.Dataset()
        dset["time"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"))

    np.random.seed(seed)

    ntimes = len(dset["time"])

    dset["z_i"] = xr.DataArray(
        np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]), dims=("z_i")
    )
    dset["z_l"] = xr.DataArray(
        np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]), dims=("z_l")
    )
    dset["xh"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="xh")
    dset["yh"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="yh")

    dset["thetao"] = xr.DataArray(
        np.random.normal(15.0, 5.0, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
    )
    dset["so"] = xr.DataArray(
        np.random.normal(35.0, 1.5, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
    )
    dset["volcello"] = xr.DataArray(
        np.random.normal(1000.0, 100.0, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
    )

    deptho = np.array(
        [
            np.random.uniform(0.0, 5.0, 5),
            np.random.uniform(0.0, 15.0, 5),
            np.random.uniform(0.0, 185.0, 5),
            np.random.uniform(0.0, 1815.0, 5),
            np.random.uniform(0.0, 6185.0, 5),
        ]
    )

    dset["deptho"] = xr.DataArray(deptho, dims=({"yh": dset.yh, "xh": dset.xh}))

    areacello = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)), dims=({"yh": dset.yh, "xh": dset.xh})
    )
    areacello = areacello / areacello.sum()
    dset["areacello"] = areacello * 3.6111092e14

    return dset


def generate_test_data_dz(seed=123):
    """Function to generate test dataset for partial bottom cells

    Parameters
    ----------
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
    """
    xh = np.arange(1, 6)
    xh = xr.DataArray(xh, dims=("xh"))

    yh = np.arange(10, 60, 10)
    yh = xr.DataArray(yh, dims=("yh"))

    np.random.seed(seed)

    deptho = np.random.uniform(0.0, 100.0, (5, 5))
    deptho[2, 2] = np.nan
    deptho[2, 3] = np.nan
    deptho = xr.DataArray(deptho, coords=(yh, xh))

    z_i = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    z_i = xr.DataArray(z_i, dims=("z_i"))

    z_l = np.array((z_i[1::] + z_i[0:-1]) / 2.0)
    z_l = xr.DataArray(z_l, dims=("z_l"))

    dset = xr.Dataset({"deptho": deptho, "z_l": z_l, "z_i": z_i})

    return dset


def generate_test_data_time(start_year=1981, nyears=5, calendar="noleap", seed=123):
    """Function to generate test dataset with monthly time resolution

    Parameters
    ----------
    start_year : int, optional
        Starting year, by default 1981
    nyears : int, optional
        Number of years to generate, by default 5
    calendar : str, optional
        CF-time recognized calendar, by default "noleap"
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset of annual averages
    """

    dset = generate_time_stub(start_year=start_year, nyears=nyears, calendar=calendar)

    lon = [1.0, 2.0, 3.0, 4.0, 5.0]
    lon = xr.DataArray(lon, {"lon": lon})

    lat = [1.0, 2.0, 3.0, 4.0, 5.0]
    lat = xr.DataArray(lat, {"lat": lat})

    np.random.seed(seed)
    dset["var_a"] = xr.DataArray(
        np.random.normal(100, 20, (60, 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": dset.time, "lat": lat, "lon": lon},
    )

    np.random.seed(seed * 2)
    dset["var_b"] = xr.DataArray(
        np.random.normal(100, 20, (60, 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": dset.time, "lat": lat, "lon": lon},
    )

    return dset


def generate_test_data_uv(start_year=1981, nyears=0, calendar="noleap", seed=123):
    """Function to generate dataset for unit testing

    This function generates a test dataset. It includes 5 points in the
    vertical, latitudinal, and longitudinal dimensions. The dataset
    contains random values for `uo`, `vo`, `Coriolis`, and `areacello_bu`.
    The dataset also includes the `dyCv` and `dxCu` grid information.

    If nyears == 0 (default), the time coordinate is a 5 point integer
    array. If nyears >= 1, a real world monthly time axis is generated.

    Parameters
    ----------
    start_year : int, optional
        Starting year, by default 1981
    nyears : int, optional
        Number of years to generate, by default 0
    calendar : str, optional
        CF-time recognized calendar, by default "noleap"
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        ntimes x 5 x 5 x 5 point dataset for unit testing
    """

    if nyears >= 1:
        dset = generate_time_stub(
            start_year=start_year, nyears=nyears, calendar=calendar
        )
    else:
        dset = xr.Dataset()
        dset["time"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"))

    np.random.seed(seed)

    ntimes = len(dset["time"])

    dset["z_i"] = xr.DataArray(
        np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]), dims=("z_i")
    )
    dset["z_l"] = xr.DataArray(
        np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]), dims=("z_l")
    )
    dset["xh"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="xh")
    dset["xq"] = xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.5], dims="xq")
    dset["yh"] = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims="yh")
    dset["yq"] = xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.5], dims="yq")

    dset["uo"] = xr.DataArray(
        np.random.normal(0.0061, 0.08, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xq": dset.xq}),
    )
    dset["vo"] = xr.DataArray(
        np.random.normal(0.00077, 0.04, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yq": dset.yq, "xh": dset.xh}),
    )
    dset["dxCu"] = xr.DataArray(np.ones((5, 5)), dims=({"yh": dset.yh, "xq": dset.xq}),)
    dset["dyCv"] = xr.DataArray(np.ones((5, 5)), dims=({"yq": dset.yq, "xh": dset.xh}),)
    dset["Coriolis"] = xr.DataArray(
        np.random.normal(1.21e-5, 0.00011, (5, 5)),
        dims=({"yq": dset.yq, "xq": dset.xq}),
    )
    areacello_bu = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)), dims=({"yq": dset.yh, "xq": dset.xh})
    )
    areacello_bu = areacello_bu / areacello_bu.sum()
    dset["areacello_bu"] = areacello_bu * 3.6111092e14

    return dset


def generate_time_stub(start_year=1981, nyears=5, calendar="noleap"):
    """Function to generate a dataset with a time coordinate

    This function creates a "stub" dataset that can be used a starting
    point for further test datasets.  It returns a cf-time index and
    related FMS bounds and helper fields. Monthly values for `nyears`
    are generated.

    Parameters
    ----------
    start_year : int
        Starting year for time coordinate, by default 1981
    nyears : int
        Number of years of monthly data to generate, by default 5
    calendar : str
        Cf-time recognized calendar, by default "noleap"

    Returns
    -------
    xarray.core.dataset.Dataset
        Stub dataset with time coordinate
    """
    bounds = xr.cftime_range(
        f"{start_year}-01-01", freq="MS", periods=(nyears * 12) + 1, calendar=calendar
    )
    time_bnds = list(zip(bounds[0:-1], bounds[1::]))
    bnds = np.array([1, 2])

    time = [x[0] + (x[1] - x[0]) / 2 for x in time_bnds]
    time = xr.DataArray(time, {"time": time})

    time_bnds = xr.DataArray(time_bnds, {"time": time, "bnds": bnds})

    average_T1 = xr.DataArray(time_bnds[:, 0].values, {"time": time})
    average_T2 = xr.DataArray(time_bnds[:, 1].values, {"time": time})
    average_DT = average_T2 - average_T1

    return xr.Dataset(
        {
            "time": time,
            "time_bnds": time_bnds,
            "average_T1": average_T1,
            "average_T2": average_T2,
            "average_DT": average_DT,
        }
    )
