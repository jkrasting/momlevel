import numpy as np
import xarray as xr
import momlevel.eos as eos


def thermosteric(
    dset,
    reference=None,
    temp_varname="thetao",
    salt_varname="so",
    vol_varname="volcello",
    rho_varname="rho",
    area_varname="areacello",
    vert_coord="z_l",
    time_dim="time",
    equation_of_state="wright",
):
    """Calculates the thermosteric sea level change and other
    associated quantities relative to a known reference state.

    This routine isolates the contributions of temperature change
    on in-situ density by using a constant reference salinity
    value.

    An expansion coefficient is computed that relates the
    temperature-modulated in-situ density to a reference density
    value. This expansion coefficient is multiplied by a
    reference cell volume and divided by the cell area to yield
    an inferred change in cell height. The sum of changes in the
    vertical corresponds to the column thermosteric sea level change.

    Parameters
    ----------
    dset : xarray.Dataset
        Input dataset containing timeseries of 3-dimensional
        temperature [volcello], salinity[so], volume [volcello],
        and cell area [areacello]
    reference : xarray.Dataset, optional
        Time-invariant (i.e. single record) of 3-dimensional
        salinity, volume, and density [rho] to use as the
        reference state. If not supplied, the first time record
        of the input dataset is used as the reference state,
        by default None
    temp_varname : str, optional
        Temperature variable name, by default "thetao"
    salt_varname : str, optional
        Salinity variable name, by default "so"
    vol_varname : str, optional
        Cell volume variable name, by default "volcello"
    rho_varname : str, optional
        In-situ density variable name, by default "rho"
    area_varname : str, optional
        Cell area variable name, by default "areacello"
    vert_coord : str, optional
        Vertical coordinate name, by default "z_l"
    time_dim : str, optional
        Time dimension name, by default "time"
    equation_of_state : str, optional
        Equation of state of sea water to use for calculating
        in-situ density, by default "wright"

    Returns
    -------
    xarray.Dataset

    """

    # get cell area
    cell_area = dset[area_varname]

    # figure out what to use for the reference fields
    if reference is None:
        reference_so = None
        reference_vol = None
        reference_rho = None
    else:
        reference_so = (
            reference[salt_varname] if salt_varname in reference.variables else None
        )

        reference_vol = (
            reference[vol_varname] if vol_varname in reference.variables else None
        )

        reference_rho = (
            reference[rho_varname] if rho_varname in reference.variables else None
        )

        if any(x is None for x in [reference_so, reference_vol, reference_rho]):
            raise RuntimeError("Supplied reference dataset is incomplete")

    # use reference volume and salinity fields from existing dataset if not defined
    reference_vol = (
        dset[vol_varname].isel({time_dim: 0})
        if reference_vol is None
        else reference_vol
    )
    reference_so = (
        dset[salt_varname].isel({time_dim: 0}) if reference_so is None else reference_so
    )

    # remove any singleton dimensions
    reference_vol = reference_vol.squeeze()
    reference_so = reference_so.squeeze()

    # sanity check that a time dimension is not present
    assert (
        time_dim not in reference_vol.dims
    ), "Reference volume is a state, it cannot contain a time dimension"
    assert (
        time_dim not in reference_so.dims
    ), "Reference salinity is a state, it cannot contain a time dimension"

    # determine the equation of state to use
    equation_of_state = eos.__dict__[equation_of_state]

    # approximate pressure from depth coordinate
    vertical_coord = dset[vert_coord] * 1.0e4

    # calculate reference density if it is not present
    if reference_rho is None:
        reference_rho = xr.apply_ufunc(
            equation_of_state,
            dset[temp_varname].isel({time_dim: 0}).squeeze(),
            reference_so,
            vertical_coord,
            dask="allowed",
        )
        reference_rho.attrs = {
            "long_name": "Reference in-situ density",
            "units": "kg m-3",
        }

    assert (
        time_dim not in reference_rho.dims
    ), "Reference density is a state, it cannot contain a time dimension"

    # calculate reference height for each grid cell
    reference_height = reference_vol / cell_area
    reference_height.attrs = {"long_name": "Reference column height", "units": "m"}

    # calculate density as function of only changing temperature; hold salinity
    # constant at the reference value
    rho = xr.apply_ufunc(
        equation_of_state,
        dset[temp_varname],
        reference_so,
        vertical_coord,
        dask="allowed",
    )

    # calculate the expansion coefficient for each grid cell
    expansion_coeff = np.log(reference_rho / rho)
    expansion_coeff = expansion_coeff.transpose(*(time_dim, ...))
    expansion_coeff.attrs = {"long_name": "Expansion coefficient"}

    # calculate thermosteric sl
    thermosteric = (reference_height * expansion_coeff).sum(dim=vert_coord)
    thermosteric = thermosteric.transpose(*(time_dim, ...))
    thermosteric.attrs = {"long_name": "Thermosteric column height", "units": "m"}

    # return an Xarray Dataset with the results
    result = xr.Dataset()
    result["reference_so"] = reference_so
    result["reference_vol"] = reference_vol
    result["reference_rho"] = reference_rho.where(reference_vol.notnull())
    result["reference_height"] = reference_height.where(reference_vol.notnull())
    result["expansion_coeff"] = expansion_coeff.where(reference_vol.notnull())
    result["thermosteric"] = thermosteric.where(
        reference_vol.isel({vert_coord: 0}).notnull()
    )

    return result
