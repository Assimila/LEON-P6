"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

from openeo.udf import inspect
import xarray as xr


LOG_CODE = "idxmin_t"


class Context(TypedDict):
    pass


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Find the idxmin of dimension "t".
    Convert from datetime to decimal year (float)

    Arguments:
        cube: xarray DataArray.
            dims (t, ...) where "..." indicates other broadcast-able dimensions.
        context: user-provided arguments.
    """
    inspect(
        data=cube.sizes, message="Input cube dimensions", code=LOG_CODE, level="debug"
    )
    nan_pct = (cube.isnull().sum().item() / cube.size) * 100
    inspect(
        message=f"NaNs in input cube = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )
    inspect(
        message=f"Size of cube = {format_bytes(cube.nbytes)}",
        code=LOG_CODE,
        level="debug",
    )

    # Get the datetime coordinates for the minimums
    min_times = cube.idxmin(dim="t")
    # this is now a cube of dtype datetime64

    # not available in xarray 2024.7
    # decimal_year = min_times.dt.decimal_year

    year = min_times.dt.year
    doy = min_times.dt.dayofyear
    days_in_year = xr.where(min_times.dt.is_leap_year, 366, 365)

    decimal_year = year + (doy - 1) / days_in_year

    return decimal_year
