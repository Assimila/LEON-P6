"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

import xarray as xr

from openeo.udf import inspect

LOG_CODE = "pixel_area_ha"

# 1 ha = 10000 m^2
M2_PER_HA = 10000


class Context(TypedDict):
    # I can't find a way to access pixel size (spatial resolution) within the UDF

    # spatial resolution / linear pixel size (units: m)
    spatial_resolution: float


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Convert a pixel mask to pixel area (units: ha), based on spatial resolution.

    Why not just use native openEO processes?
    Because there seems to be a bug related to Parameters.
    https://forum.dataspace.copernicus.eu/t/udp-parameter-not-applied/5282/3

    Arguments:
        cube: xarray DataArray
            A pixel mask (0 / 1). Values are multiplied by the pixel area.
        context: user-provided arguments.
    """
    spatial_resolution = context["spatial_resolution"]
    if not isinstance(spatial_resolution, float | int):
        raise TypeError("spatial_resolution should be a float")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution should be positive")

    pixel_area_ha = spatial_resolution * spatial_resolution / M2_PER_HA

    inspect(
        message=(
            f"converting mask to area: spatial_resolution = {spatial_resolution} m, "
            f"pixel_area = {pixel_area_ha} ha"
        ),
        code=LOG_CODE,
        level="debug",
    )

    return cube * pixel_area_ha
