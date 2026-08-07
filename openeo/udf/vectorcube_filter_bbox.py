"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

import typing

import geopandas as gpd
import shapely
import xarray as xr
from pyproj import CRS, Transformer
from shapely.ops import transform

from openeo.udf import inspect

LOG_CODE = "vectorcube_filter_bbox"


class BBox(typing.TypedDict):
    west: float
    south: float
    east: float
    north: float
    crs: typing.NotRequired[str]  # default: EPSG:4326


class Context(typing.TypedDict):
    # see spatial_extent in https://openeo.org/documentation/1.0/processes.html#load_collection
    spatial_extent: BBox


def apply_vectorcube(
    geometries: gpd.GeoDataFrame, cube: xr.DataArray, context: Context
) -> tuple[gpd.GeoDataFrame, xr.DataArray]:
    """
    Filter out geometries that are not completely contained within the spatial extent.

    Arguments:
        geometries: input geometries as a geopandas.GeoDataFrame.
            This contains the actual shapely geometries and optional properties.
        cube: a data cube with dimensions (geometries, time, bands) where time and bands are optional.
            The dimension "geometries" may be called "geometry".
            The coordinates for the geometry dimension are integers
            and match the index of the geometries in the geometries parameter.
        context: user-provided arguments.
    """
    n_geoms = len(geometries)
    inspect(message=f"Number of gpd rows = {n_geoms}", code=LOG_CODE, level="info")
    inspect(
        data=cube.sizes, message="Input cube dimensions", code=LOG_CODE, level="info"
    )

    # parse and validate context
    spatial_extent = context["spatial_extent"]
    bbox_crs = spatial_extent.get("crs", "EPSG:4326")
    bbox = shapely.geometry.box(
        minx=spatial_extent["west"],
        miny=spatial_extent["south"],
        maxx=spatial_extent["east"],
        maxy=spatial_extent["north"],
    )

    # handle CRS conversion to the CRS of the geopandas dataframe, if necessary
    source_crs = CRS.from_user_input(bbox_crs)
    target_crs = geometries.crs or CRS.from_epsg(4326)

    if not source_crs.equals(target_crs):
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        bbox = transform(transformer.transform, bbox)

    keep = geometries.geometry.within(bbox)  # boolean Series

    n_keep = sum(keep)
    inspect(
        message=f"Keeping {n_keep} of {n_geoms} geometries within bbox",
        code=LOG_CODE,
        level="info",
    )

    geometries = geometries.loc[keep]

    # now remove an data from `cube` for the geometries that we dropped

    if "geometry" in cube.dims:
        geom_dim = "geometry"
    elif "geometries" in cube.dims:
        geom_dim = "geometries"
    else:
        raise ValueError(f"Cube has no geometry dimension; dims={cube.dims}")

    cube = cube.sel({geom_dim: geometries.index})

    return geometries, cube
