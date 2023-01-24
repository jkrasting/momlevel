""" time.py - time module for generating test data """

import datetime as dt

import cftime
import numpy as np
import xarray as xr

__all__ = [
    "generate_daily_timeaxis",
    "generate_time_stub",
]


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
