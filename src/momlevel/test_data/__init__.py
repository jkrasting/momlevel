""" test_data - module for generating test data """

import numpy as np
import xarray as xr

from .time import generate_time_stub
from .tripolar import xy_fields, zlevel_fields

__all__ = [
    "generate_test_data",
    "generate_test_data_dz",
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

    dset = xy_fields(dset)
    dset = zlevel_fields(dset)

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

    dset = xy_fields(dset)
    dset = zlevel_fields(dset)

    dset["xq"] = xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.5], dims="xq")
    dset["yq"] = xr.DataArray([1.5, 2.5, 3.5, 4.5, 5.5], dims="yq")

    dset["uo"] = xr.DataArray(
        np.random.default_rng(seed).normal(0.0061, 0.08, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yh": dset.yh, "xq": dset.xq}),
        attrs={
            "long_name": "Sea Water X Velocity",
            "units": "m s-1",
            "standard_name": "sea_water_x_velocity",
            "interp_method": "none",
            "cell_methods": "z_l:mean yh:mean xq:point time: mean",
            "time_avg_info": "average_T1,average_T2,average_DT",
        },
    )
    dset["vo"] = xr.DataArray(
        np.random.default_rng(seed).normal(0.00077, 0.04, (ntimes, 5, 5, 5)),
        dims=({"time": dset.time, "z_l": dset.z_l, "yq": dset.yq, "xh": dset.xh}),
        attrs={
            "long_name": "Sea Water Y Velocity",
            "units": "m s-1",
            "standard_name": "sea_water_y_velocity",
            "interp_method": "none",
            "cell_methods": "z_l:mean yq:point xh:mean time: mean",
            "time_avg_info": "average_T1,average_T2,average_DT",
        },
    )
    dset["dxCu"] = xr.DataArray(
        np.ones((5, 5)),
        dims=({"yh": dset.yh, "xq": dset.xq}),
        attrs={
            "long_name": "Delta(x) at u points (meter)",
            "units": "m",
            "cell_methods": "time: point",
            "interp_method": "none",
        },
    )
    dset["dyCv"] = xr.DataArray(
        np.ones((5, 5)),
        dims=({"yq": dset.yq, "xh": dset.xh}),
        attrs={
            "long_name": "Delta(y) at v points (meter)",
            "units": "m",
            "cell_methods": "time: point",
            "interp_method": "none",
        },
    )
    dset["Coriolis"] = xr.DataArray(
        np.random.default_rng(seed).normal(1.21e-5, 0.00011, (5, 5)),
        dims=({"yq": dset.yq, "xq": dset.xq}),
        attrs={
            "long_name": "Coriolis parameter at corner (Bu) points",
            "units": "s-1",
            "cell_methods": "time: point",
            "interp_method": "none",
        },
    )
    areacello_bu = xr.DataArray(
        np.random.default_rng(seed).normal(100.0, 10.0, (5, 5)),
        dims=({"yq": dset.yh, "xq": dset.xh}),
    )
    areacello_bu = areacello_bu / areacello_bu.sum()
    dset["areacello_bu"] = areacello_bu * 3.6111092e14

    dset["areacello_bu"].attrs = {
        "long_name": "Ocean Grid-Cell Area",
        "units": "m2",
        "cell_methods": "area:sum yq:sum xq:sum time: point",
        "standard_name": "cell_area",
    }

    return dset
