""" test_data.py - module for generating test data """

import xarray as xr
import numpy as np

__all__ = ["generate_test_data", "generate_test_data_dz", "generate_test_data_time"]


def generate_test_data(seed=123):
    """Function to generate dataset for unit testing

    Parameters
    ----------
    seed : int, optional
        Random number generator seed. By default, 123

    Returns
    -------
    xarray.core.dataset.Dataset
        5x5x5x5 point dataset for unit testing
    """
    np.random.seed(seed)
    time = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=("time"))
    z_i = xr.DataArray(np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]), dims=("z_i"))
    z_l = xr.DataArray(np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]), dims=("z_l"))
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

    deptho = np.array(
        [
            np.random.uniform(0.0, 5.0, 5),
            np.random.uniform(0.0, 15.0, 5),
            np.random.uniform(0.0, 185.0, 5),
            np.random.uniform(0.0, 1815.0, 5),
            np.random.uniform(0.0, 6185.0, 5),
        ]
    )

    deptho = xr.DataArray(deptho, dims=({"yh": yh, "xh": xh}))
    areacello = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)), dims=({"yh": yh, "xh": xh})
    )
    areacello = areacello / areacello.sum()
    areacello = areacello * 3.6111092e14

    dset = xr.Dataset(
        {
            "thetao": thetao,
            "so": so,
            "volcello": volcello,
            "areacello": areacello,
            "deptho": deptho,
        }
    )
    dset = dset.assign_coords(
        {"time": time, "z_l": z_l, "z_i": z_i, "yh": yh, "xh": xh}
    )

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

    lon = [1.0, 2.0, 3.0, 4.0, 5.0]
    lon = xr.DataArray(lon, {"lon": lon})

    lat = [1.0, 2.0, 3.0, 4.0, 5.0]
    lat = xr.DataArray(lat, {"lat": lat})

    np.random.seed(seed)
    var_a = xr.DataArray(
        np.random.normal(100, 20, (60, 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": time, "lat": lat, "lon": lon},
    )

    np.random.seed(seed * 2)
    var_b = xr.DataArray(
        np.random.normal(100, 20, (60, 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": time, "lat": lat, "lon": lon},
    )

    return xr.Dataset(
        {
            "time": time,
            "lon": lon,
            "lat": lat,
            "time_bnds": time_bnds,
            "average_T1": average_T1,
            "average_T2": average_T2,
            "average_DT": average_DT,
            "var_a": var_a,
            "var_b": var_b,
        }
    )
