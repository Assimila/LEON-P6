"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

from openeo.udf import inspect
import xarray as xr


LOG_CODE = "multitemporal_speckle_filter"


class Context(TypedDict):
    # the radius of the circular filter in pixels.
    # Should be at least 1.
    # for example, radius=2 means that the filter will consider a 5x5 window.
    radius: int

    # size of the temporal window (number of time steps to consider).
    # Should be at least 3, and an odd number.
    window_size: int


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Apply a multitemporal speckle filter.
    With a moving window.

    Reference: https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.sar.processing.ui/operators/MultiTemporalSpeckleFilterOp.html

    There is no special case for edge pixels.
    Because we expect to be processing a buffered chunk of a larger image,
    with an overlap of at least `radius` pixels on each side.

    Arguments:
        cube: xarray Dataset
            Should have spatial dimensions "x" and "y" of size at least (2 * `radius` + 1).
            And temporal dimension "t" of size at least `window`.
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

    radius = context["radius"]
    if not isinstance(radius, int):
        raise ValueError("radius should be an integer")
    if radius < 1:
        raise ValueError("radius should be at least 1")
    window_size = context["window_size"]
    if not isinstance(window_size, int):
        raise ValueError("window should be an integer")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window should be an odd integer, at least 3")

    if cube.sizes["t"] < window_size:
        raise ValueError("Insufficient temporal data")

    # circular kernel over dims (x_win, y_win)

    kernel = xr.DataArray(
        1.0,
        dims=["x_win", "y_win"],
        coords={
            "x_win": range(-radius, radius + 1),
            "y_win": range(-radius, radius + 1),
        },
    )
    kernel = kernel.where(kernel.x_win**2 + kernel.y_win**2 <= radius**2, 0.0)
    kernel = kernel / kernel.sum()

    # construct a rolling window over (x, y)
    # adding dimensions (x_win, y_win) to the cube

    window = cube.rolling(
        x=2 * radius + 1,
        y=2 * radius + 1,
        # the window is centred on the pixel of interest in (x, y)
        center=True,
    ).construct(
        x="x_win",
        y="y_win",
    )

    # local mean

    mu = window.dot(kernel)

    # numerator - normalised intensity

    eps = 1e-12
    if (abs(mu) < eps).any():
        inspect(
            message="Warning: local mean is close to zero for some pixels, which may lead to instability in the filter.",
            code=LOG_CODE,
            level="warning",
        )

    ni = cube / mu

    # rolling window over time dimension

    sum_ni = ni.rolling(t=window_size, center=True).sum()

    output = mu / window_size * sum_ni

    nan_pct = (output.isnull().sum().item() / output.size) * 100
    inspect(
        message=f"NaNs in returned cube = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )

    return output
