#!/usr/bin/env python
"""
create_climatology.py

Created by Cameron Cummins

Description:
    For implementation into CUPiD, modeled after ADF
    'scripts/averaging/create_climo_files.py'. The create_climatology function
    is a wrapper function for compute_climatology, it just handles the input
    as strings with a few more useful checks to make it easier to implement
    upstream. The xarray functions are inherently compatable with Dask, so I
    think introducing multiprocessing or parallel arguments here is premature.
    Dataset opening is handled by xarray.open_mfdataset regardless of how many
    files are specified. This will return a Dask array that can be computed in
    parallel or forced into serial.

Contact: cameron.cummins@utexas.edu
Date: 6/3/2024
"""
import xarray
import cftime


def create_climatology(timeseries_path: str, var_name: str,
                       time_dim: str="time", start_time: str=None, end_time: str=None, date_format: str=None) -> xarray.Dataset:
    """
    Creates climatology dataset for a given variable timeseries with some input parameter checking.

    Parameters
    ----------
        timeseries_path : str, required
            String glob providing path to timeseries file(s). Can use wildcard (*).
        var_name : str, required
            Name of variable stored in timeseries file(s).
        time_dim : str, optional
            Name of time dimension. Default: 'time'
        start_time : str, optional*
            Starting date for period to average over. Defaults to the first timestep of the timeseries.
        end_time : str, optional*
            Ending date for period to average over. Defaults to the last timestep of the timeseries.
        date_format : str, optional*
            If either a start of end date are specified, a date format string is required in order
            to parse the date strings. Use datetime format codes.
            https://docs.python.org/3/library/datetime.html#format-codes)

    Returns
    -------
        climatology_ds : xarray.Dataset
            Dataset containing the monthly and seasonal averages over the specified time period for the
            given timeseries. Attributes are preserved.
    """
    if "*" not in timeseries_path:
        timeseries_path = [timeseries_path]
    if date_format is None and (start_time is not None or end_time is not None):
        raise ValueError("start_time/end_time string requires format codes (see https://docs.python.org/3/library/datetime.html#format-codes).")

    timeseries = xarray.open_mfdataset(timeseries_path, decode_times=True)
    if var_name not in timeseries:
        raise ValueError(f"var_name '{var_name}' not found in dataset at {timeseries_path}.")

    start_cftime = timeseries[time_dim].values[0]
    end_cftime = timeseries[time_dim].values[-1]
    timeseries_calendar = timeseries.time.values[0].calendar

    if start_time is not None:
        start_cftime = cftime.datetime.strptime(start_time, date_format, calendar=timeseries_calendar)
    if end_time is not None:
        end_cftime = cftime.datetime.strptime(end_time, date_format, calendar=timeseries_calendar)

    if (end_cftime - start_cftime).days < 30*365 and not (timeseries_calendar == "360_day" and (end_cftime - start_cftime).days < 30*360):
        print(f"Warning: creating climatology with less than 30 years of data from {start_cftime} to {end_cftime}")

    timeseries_var = timeseries[var_name].sel(time=slice(start_cftime, end_cftime))

    climatology_ds = compute_climatology(timeseries_var, time_dim=time_dim, prefix=f"{var_name}_")
    climatology_ds.attrs = timeseries.attrs
    climatology_ds.attrs |= {
        "description": "monthly and seasonal climatology",
        "time_period": f"{start_cftime} to {end_cftime}"
    }

    return climatology_ds


def compute_climatology(timeseries: xarray.DataArray, time_dim: str="time", prefix: str="") -> xarray.Dataset:
    """
    Computes the monthly and seasonal averages for a given data array.

    Parameters
    ----------
        timeseries : xarray.DataArray, required
            Timeseries data to compute monthly averages over.
        time_dim : str, optional
            Name of time dimension to compute averages over. Default: 'time'
        prefix : str, optional
            Prefix to add to beginning of variable names in output dataset. Useful when
            working with multiple climatologies. Default is no prefix.

    Returns
    -------
        climatology_ds : xarray.Dataset
            Dataset with monthly and seasonal variables.
            Monthly indices are abbreviated for ease of use.
            MAM --> March-April-May
            JJA --> June-July-August
            SON --> September-October-November
            DJF --> December-January-February
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_means = timeseries.groupby(f"{time_dim}.month").mean(keep_attrs=True).assign_coords(dict(month=months))

    climatology = xarray.Dataset(
        {
            f"{prefix}MONTH": monthly_means,
            f"{prefix}MAM": monthly_means.sel(month=["Mar", "Apr", "May"]).mean(dim="month", keep_attrs=True),
            f"{prefix}JJA": monthly_means.sel(month=["Jun", "Jul", "Aug"]).mean(dim="month", keep_attrs=True),
            f"{prefix}SON": monthly_means.sel(month=["Sep", "Oct", "Nov"]).mean(dim="month", keep_attrs=True),
            f"{prefix}DJF": monthly_means.sel(month=["Dec", "Jan", "Feb"]).mean(dim="month", keep_attrs=True)
        }
    )
    return climatology