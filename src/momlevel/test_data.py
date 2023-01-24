""" test_data.py - module for generating test data """

import datetime as dt

import cftime
import numpy as np
import xarray as xr

__all__ = [
    "generate_daily_timeaxis",
    "generate_test_data",
    "generate_test_data_dz",
    "generate_test_data_time",
    "generate_test_data_uv",
]


def _add_h_point_fields(dset=None):

    dset = xr.Dataset if dset is None else dset

    dset["xh"] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="xh",
        attrs={
            "long_name": "h point nominal longitude",
            "units": "degrees_east",
            "axis": "X",
            "cartesian_axis": "X",
        },
    )

    dset["yh"] = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims="yh",
        attrs={
            "long_name": "h point nominal latitude",
            "units": "degrees_north",
            "axis": "Y",
            "cartesian_axis": "Y",
        },
    )

    # geolon / geolat as cell centers
    lon = np.arange(0.0, 361.0, 72.0)
    lat = np.arange(-90.0, 91.0, 36.0)

    lon = [(lon[x] + lon[x + 1]) / 2.0 for x in range(0, len(lon) - 1)]
    lat = [(lat[x] + lat[x + 1]) / 2.0 for x in range(0, len(lat) - 1)]
    geolon, geolat = np.meshgrid(lon, lat)

    dset["geolon"] = xr.DataArray(
        geolon,
        dims=("yh", "xh"),
        attrs={
            "long_name": "Longitude of tracer (T) points",
            "units": "degrees_east",
            "cell_methods": "time: point",
        },
    )

    dset["geolat"] = xr.DataArray(
        geolat,
        dims=("yh", "xh"),
        attrs={
            "long_name": "Latitude of tracer (T) points",
            "units": "degrees_north",
            "cell_methods": "time: point",
        },
    )

    areacello = xr.DataArray(
        np.random.normal(100.0, 10.0, (5, 5)),
        dims=({"yh": dset.yh, "xh": dset.xh}),
    )
    areacello = areacello / areacello.sum()
    dset["areacello"] = areacello * 3.6111092e14
    dset["areacello"].attrs = {
        "long_name": "Ocean Grid-Cell Area",
        "units": "m2",
        "cell_methods": "area:sum yh:sum xh:sum time: point",
        "standard_name": "cell_area",
    }
    return dset


def _add_z_level_fields(dset=None):
    dset = xr.Dataset() if dset is None else dset

    dset["z_i"] = xr.DataArray(
        np.array([0.0, 5.0, 15.0, 185.0, 1815.0, 6185.0]),
        dims=("z_i"),
        attrs={
            "long_name": "Depth at interface",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
        },
    )

    dset["z_l"] = xr.DataArray(
        np.array([2.5, 10.0, 100.0, 1000.0, 4000.0]),
        dims=("z_l"),
        attrs={
            "long_name": "Depth at cell center",
            "units": "meters",
            "axis": "Z",
            "positive": "down",
            "edges": "z_i",
        },
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

    dset["deptho"] = xr.DataArray(
        deptho,
        dims=({"yh": dset.yh, "xh": dset.xh}),
        attrs={
            "long_name": "Sea Floor Depth",
            "units": "m",
            "cell_methods": "area:mean yh:mean xh:mean time: point",
            "cell_measures": "area: areacello",
            "standard_name": "sea_floor_depth_below_geoid",
        },
    )

    # TO-DO: move zl, zi, and deptho in here

    return dset


def generate_daily_timeaxis(start_year=1979, nyears=2, calendar="noleap"):
    """Function to generate a daily time axis for testing

    Parameters
    ----------
    start_year : int, optional
        Starting year for timeseries, by default 1979
    nyears : int, optional
        Number of years for test data, by default 2
    calendar : str, optional
        Valid `cftime` calendar, by default "noleap"

    Returns
    -------
    List[cftime._cftime.Datetime]
        List of cftime datetime objects for specified calendar
    """

    init = cftime.datetime(start_year, 1, 1, calendar=calendar)
    endtime = cftime.datetime(start_year + nyears, 1, 1, calendar=calendar)

    days = [init + dt.timedelta(days=x) for x in range(0, 366 * nyears)]
    days = [x for x in days if x < endtime]

    return days


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
        dset["time"] = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=("time"),
            attrs={
                "long_name": "time",
                "cartesian_axis": "T",
                "calendar_type": calendar,
                "bounds": "time_bnds",
            },
        )

    ntimes = len(dset["time"])

    dset = _add_h_point_fields(dset)
    dset = _add_z_level_fields(dset)

    dset["thetao"] = xr.DataArray(
        np.random.default_rng(seed).normal(15.0, 5.0, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
        attrs={
            "long_name": "Sea Water Potential Temperature",
            "units": "degC",
            "cell_measures": "volume: volcello area: areacello",
            "standard_name": "sea_water_potential_temperature",
            "cell_methods": "area:mean z_l:mean yh:mean xh:mean time: mean",
            "time_avg_info": "average_T1,average_T2,average_DT",
        },
    )

    dset["so"] = xr.DataArray(
        np.random.default_rng(seed).normal(35.0, 1.5, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
        attrs={
            "long_name": "Sea Water Salinity",
            "units": "psu",
            "cell_measures": "volume: volcello area: areacello",
            "standard_name": "sea_water_salinity",
            "cell_methods": "area:mean z_l:mean yh:mean xh:mean time: mean",
            "time_avg_info": "average_T1,average_T2,average_DT",
        },
    )

    dset["volcello"] = xr.DataArray(
        np.random.default_rng(seed).normal(1000.0, 100.0, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xh": dset.xh}),
        attrs={
            "long_name": "Ocean grid-cell volume",
            "units": "m3",
            "cell_measures": "area: areacello",
            "standard_name": "ocean_volume",
            "cell_methods": "area:sum z_l:sum yh:sum xh:sum time: mean",
            "time_avg_info": "average_T1,average_T2,average_DT",
        },
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

    deptho = np.random.default_rng(seed).uniform(0.0, 100.0, (5, 5))
    deptho[2, 2] = np.nan
    deptho[2, 3] = np.nan
    deptho = xr.DataArray(deptho, coords=(yh, xh))

    z_i = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    z_i = xr.DataArray(z_i, dims=("z_i"))

    z_l = np.array((z_i[1::] + z_i[0:-1]) / 2.0)
    z_l = xr.DataArray(z_l, dims=("z_l"))

    dset = xr.Dataset({"deptho": deptho, "z_l": z_l, "z_i": z_i})

    return dset


def generate_test_data_time(
    start_year=1981, nyears=5, calendar="noleap", seed=123, frequency="MS"
):
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
    frequency : str
        Frequency string compatible with xarray.cftime_range

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset of sample time series data
    """

    dset = generate_time_stub(
        start_year=start_year, nyears=nyears, calendar=calendar, frequency=frequency
    )

    lon = [1.0, 2.0, 3.0, 4.0, 5.0]
    lon = xr.DataArray(lon, {"lon": lon})

    lat = [1.0, 2.0, 3.0, 4.0, 5.0]
    lat = xr.DataArray(lat, {"lat": lat})

    dset["var_a"] = xr.DataArray(
        np.random.default_rng(seed).normal(100, 20, (len(dset.time), 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": dset.time, "lat": lat, "lon": lon},
        attrs={"first_attribute": "foo", "second_attribute": "bar"},
    )

    dset["var_b"] = xr.DataArray(
        np.random.default_rng(seed * 2).normal(100, 20, (len(dset.time), 5, 5)),
        dims=(("time", "lat", "lon")),
        coords={"time": dset.time, "lat": lat, "lon": lon},
        attrs={"first_attribute": "foo", "second_attribute": "bar"},
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
        np.random.default_rng(seed).normal(0.0061, 0.08, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xq": dset.xq}),
    )
    dset["vo"] = xr.DataArray(
        np.random.default_rng(seed).normal(0.00077, 0.04, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yq": dset.yq, "xh": dset.xh}),
    )
    dset["dxCu"] = xr.DataArray(
        np.ones((5, 5)),
        dims=({"yh": dset.yh, "xq": dset.xq}),
    )
    dset["dyCv"] = xr.DataArray(
        np.ones((5, 5)),
        dims=({"yq": dset.yq, "xh": dset.xh}),
    )
    dset["Coriolis"] = xr.DataArray(
        np.random.default_rng(seed).normal(1.21e-5, 0.00011, (5, 5)),
        dims=({"yq": dset.yq, "xq": dset.xq}),
    )
    areacello_bu = xr.DataArray(
        np.random.default_rng(seed).normal(100.0, 10.0, (5, 5)),
        dims=({"yq": dset.yh, "xq": dset.xh}),
    )
    areacello_bu = areacello_bu / areacello_bu.sum()
    dset["areacello_bu"] = areacello_bu * 3.6111092e14

    return dset


def generate_time_stub(start_year=1981, nyears=5, calendar="noleap", frequency="MS"):
    """Function to generate a dataset with a time coordinate

    This function creates a "stub" dataset that can be used a starting
    point for further test datasets.  It returns a cf-time index and
    related FMS bounds and helper fields. Values for `nyears` are generated.

    Parameters
    ----------
    start_year : int
        Starting year for time coordinate, by default 1981
    nyears : int
        Number of years of monthly data to generate, by default 5
    calendar : str
        Cf-time recognized calendar, by default "noleap"
    frequency : str
        Frequency string compatible with xarray.cftime_range

    Returns
    -------
    xarray.core.dataset.Dataset
        Stub dataset with time coordinate
    """

    if frequency == "MS":
        bounds = xr.cftime_range(
            f"{start_year}-01-01",
            freq=frequency,
            periods=(nyears * 12) + 1,
            calendar=calendar,
        )
        bounds = bounds.values

    elif frequency == "D":
        bounds = xr.cftime_range(
            f"{start_year}-01-01",
            freq=frequency,
            periods=(nyears * 366) + 1,
            calendar=calendar,
        )
        bounds = [
            x
            for x in bounds.values
            if x <= cftime.datetime(start_year + nyears, 1, 1, calendar=calendar)
        ]

    else:
        raise ValueError(f"Time frequency '{frequency}' is not currently supported.")

    time_bnds = list(zip(bounds[0:-1], bounds[1::]))
    bnds = np.array([1, 2])

    time = [x[0] + (x[1] - x[0]) / 2 for x in time_bnds]
    time = xr.DataArray(
        time,
        {"time": time},
        attrs={
            "long_name": "time",
            "cartesian_axis": "T",
            "calendar_type": calendar,
            "bounds": "time_bnds",
        },
    )

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
