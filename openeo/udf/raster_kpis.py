"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata

from openeo.udf import inspect

LOG_CODE = "raster_kpis"

# 1 ha = 10000 m^2
M2_PER_HA = 10000


class Context(TypedDict):
    # calendar years to emit deforestation / forest-loss-to-cropland bands for
    years: list[int]

    # I can't find a way to access pixel size (spatial resolution) within the UDF

    # spatial resolution / linear pixel size (units: m)
    spatial_resolution: float


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def parse_years(years) -> list[int]:
    if not isinstance(years, list | tuple):
        raise TypeError("years should be a list")
    if len(years) == 0:
        raise ValueError("years should not be empty")

    for year in years:
        if not isinstance(year, int):
            raise TypeError(f"Invalid year: {year}")

    if len(years) != len(set(years)):
        raise ValueError("years should be unique")

    return years


def parse_spatial_resolution(spatial_resolution) -> float:
    if not isinstance(spatial_resolution, float | int):
        raise TypeError("spatial_resolution should be a float")
    if spatial_resolution <= 0:
        raise ValueError("spatial_resolution should be positive")
    return float(spatial_resolution)


def output_band_names(years: list[int]) -> list[str]:
    names = ["forest_stock_baseline_ha"]
    for y in years:
        names.append(f"deforestation_{y}_ha")
        names.append(f"forest_loss_to_cropland_{y}_ha")
    return names


def as_mask(mask: xr.DataArray) -> xr.DataArray:
    """
    Truthy finite = on. NaN and 0 are off.
    """
    return np.isfinite(mask) & (mask != 0)


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Convert year-of-deforestation, forest baseline, and cropland masks into
    raster KPI layers (units: ha).

    The input cube must have 3 bands labeled:
        - "year_of_deforestation": decimal year
        - "forest_baseline": 0/1 mask (NaN treated as 0)
        - "cropland": 0/1 mask (NaN treated as 0)

    Arguments:
        cube: xarray DataArray
            Should have a "bands" dimension with the labels above.
        context: user-provided arguments.

    Returns:
        DataArray with bands:
            forest_stock_baseline_ha,
            deforestation_{year}_ha for each context year,
            forest_loss_to_cropland_{year}_ha for each context year.
    """
    inspect(
        data=cube.sizes, message="Input cube dimensions", code=LOG_CODE, level="debug"
    )
    inspect(
        message=f"Size of cube = {format_bytes(cube.nbytes)}",
        code=LOG_CODE,
        level="debug",
    )

    years = parse_years(context["years"])
    spatial_resolution = parse_spatial_resolution(context["spatial_resolution"])
    pixel_area_ha = spatial_resolution * spatial_resolution / M2_PER_HA

    inspect(
        message=(
            f"years = {years}, spatial_resolution = {spatial_resolution} m, "
            f"pixel_area = {pixel_area_ha} ha"
        ),
        code=LOG_CODE,
        level="debug",
    )

    if "bands" not in cube.dims:
        raise ValueError("cube must have a bands dimension")


    year_of_deforestation = cube.sel(bands="year_of_deforestation", drop=True)
    forest_baseline = cube.sel(bands="forest_baseline", drop=True)
    forest_baseline_mask = as_mask(forest_baseline)
    cropland = cube.sel(bands="cropland", drop=True)
    cropland_mask = as_mask(cropland)

    layers: list[xr.DataArray] = [forest_baseline_mask]
    for y in years:
        deforestation_y = (year_of_deforestation >= y) & (year_of_deforestation < (y + 1))
        layers.append(deforestation_y)
        forest_loss_to_cropland_y = (deforestation_y & cropland_mask)
        layers.append(forest_loss_to_cropland_y)

    names = output_band_names(years)
    return xr.concat(layers, dim="bands").assign_coords(bands=names) * pixel_area_ha


def apply_metadata(metadata: CollectionMetadata, context: Context) -> CollectionMetadata:
    """
    Returns the expected cube metadata, after applying this UDF, based on input metadata.
    """
    years = parse_years(context["years"])
    names = output_band_names(years)
    # source=None replaces the whole band list, including a change of length.
    return metadata.rename_labels(dimension="bands", target=names)
