"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

from openeo.udf import inspect
import xarray as xr


LOG_CODE = "lee_speckle_filter"


class Context(TypedDict):
    # the radius of the circular filter in pixels. Should be at least 1.
    # for example, radius=2 means that the filter will consider a 5x5 window.
    radius: int

    # coefficient of variation of the noise.
    # For a look-count of N, this ~ 1 / sqrt(N)
    cv_noise: float


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Apply a circular Lee filter.

    There is no special case for edge pixels.
    Because we expect to be processing a buffered chunk of a larger image,
    with an overlap of at least `radius` pixels on each side.

    Arguments:
        cube: xarray Dataset
            Should have spatial dimensions "x" and "y" of size at least (2 * radius + 1).
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

    cv_noise = context["cv_noise"]
    if not isinstance(cv_noise, float):
        raise ValueError("cv_noise should be a number")
    if cv_noise <= 0:
        raise ValueError("cv_noise should be positive")

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

    # local mean and variance

    mu = window.dot(kernel)
    sigma2 = (window**2).dot(kernel) - mu**2

    # avoid division by zero in homogeneous areas where local variance is ~0
    eps = 1e-12
    sigma2 = sigma2.where(sigma2 > eps, eps)
    W = 1 - cv_noise**2 * mu**2 / sigma2
    W = W.clip(min=0.0, max=1.0)

    output = mu + W * (cube - mu)

    nan_pct = (output.isnull().sum().item() / output.size) * 100
    inspect(
        message=f"NaNs in returned cube = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )

    return output
