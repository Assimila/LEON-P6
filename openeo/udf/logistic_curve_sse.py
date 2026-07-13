"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

import numpy as np
from openeo.udf import inspect
import xarray as xr


LOG_CODE = "logistic_curve_sse"


class Context(TypedDict):
    # size of the temporal window (number of time steps to consider).
    # Should be at least 3, and an odd number.
    window_size: int

    # ATBD says = -2.0
    # which equates to a significant change over 3 images
    steepness_parameter: float


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Apply logistic curve as window function over dim "t".
    This is kind of like a convolution, where we calculate sum squared error at each time.

    Arguments:
        cube: xarray Dataset.
            dims (t, ...) where "..." indicates other broadcast-able dimensions.
            Temporal dimension "t" of size at least `window`.

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

    # parse and validate context

    window_size = context["window_size"]
    if not isinstance(window_size, int):
        raise ValueError("window should be an integer")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window should be an odd integer, at least 3")
    steepness_parameter = context["steepness_parameter"]
    if not isinstance(steepness_parameter, float):
        raise ValueError("steepness_parameter should be a float")

    if cube.sizes["t"] < window_size:
        raise ValueError("Insufficient temporal data")

    # compute p05 and p95
    # these have dims (...)

    # .quantile() adds a non-dimension coordinate "quantile", which we drop
    p05 = cube.quantile(0.05, dim="t").drop_vars("quantile")
    p95 = cube.quantile(0.95, dim="t").drop_vars("quantile")

    # amplitude
    k = p95 - p05

    # construct a rolling window over t
    # adding dimension t_win to the cube

    windows = cube.rolling(t=window_size, center=True).construct(
        t="t_win"
    )  # dims (t, t_win, ...)

    # window indices [0, 1, 2, 3, ...]
    t_i = windows.t_win  # dims (t_win)

    # midpoint of window
    t_0 = window_size // 2

    exponent = -1.0 * steepness_parameter * (t_i - t_0)  # dims (t_win)

    predicted = p05 + k / (1 + np.exp(exponent))  # dims (t_win, ...)

    diff = windows - predicted  # dims (t, t_win, ...)

    # reduce over t_win
    rss = (diff**2.0).sum(
        dim="t_win",
        # do not skip NaN, because rss is used as an absolute value
        skipna=False,
    )

    nan_pct = (rss.isnull().sum().item() / rss.size) * 100
    inspect(
        message=f"NaNs in returned cube = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )

    return rss
