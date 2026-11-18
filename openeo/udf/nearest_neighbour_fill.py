"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

import numpy as np
import scipy.ndimage
import xarray as xr
from openeo.metadata import CollectionMetadata

from openeo.udf import inspect

LOG_CODE = "nearest_neighbour_fill"


class Context(TypedDict):
    pass


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def nearest_neighbour_fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Arguments:
        data: numpy array with 2 dimensions only.
        mask: numpy array with 2 dimensions only.
            Truthy finite values indicate pixels that should be filled.
            NaN (nodata) is ignored - those pixels are left unchanged.
    """
    if data.ndim != 2:
        raise ValueError("data should have 2 dimensions only")
    if mask.ndim != 2:
        raise ValueError("mask should have 2 dimensions only")

    # NaN in the mask band means nodata (outside the data extent), not a fill
    # request. apply_neighborhood chunks can include overlap beyond the extent,
    # so the mask arrives with NaN edges that must be left alone.
    mask_bool = np.isfinite(mask) & (mask != 0)

    if not mask_bool.any():
        # early return
        return data.copy()

    # get a mask of finite valued pixels that can be used for replacement
    candidate_mask = np.isfinite(data) & ~mask_bool

    if not candidate_mask.any():
        # just exploding here is rather abrupt, and difficult to debug.
        # perhaps we should log a warning and fill with some fallback value?
        raise ValueError("no finite valued pixels available for replacement")

    # scipy.ndimage.distance_transform_edt computes, for each True pixel, the
    # Euclidean distance to the nearest False pixel.

    # Pass ~candidate_mask to create a lookup from every non-candidate pixel (True)
    # to the nearest candidate pixel (False).
    _, nearest_coords = scipy.ndimage.distance_transform_edt(
        ~candidate_mask, return_indices=True
    )

    # nearest_coords is a tuple of coordinate arrays, one per axis.
    # Each array has the same shape as data.
    # For pixel (i, j), the index of the nearest candidate is (II[i, j], JJ[i, j]).
    II, JJ = nearest_coords

    output = data.copy()
    # For each masked pixel, copy the value from its nearest candidate.
    output[mask_bool] = data[II[mask_bool], JJ[mask_bool]]
    return output


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Fill masked pixels with the value of the spatially nearest finite valued pixel.
    If multiple pixels are equally close, one of these is chosen.

    The input cube must have 2 bands labeled:
        - "data": input data layer
        - "mask": truthy finite = pixels to fill.
            NaN (nodata) is ignored - those pixels are left unchanged.

    If there are masked pixels, but no finite valued pixels in the data band,
    raise an error.

    Arguments:
        cube: xarray DataArray
            Should have spatial dimensions "x" and "y" and a "bands" dimension.
        context: user-provided arguments.

    Returns:
        Filled data array, dropping the "mask" band.
    """
    inspect(
        data=cube.sizes, message="Input cube dimensions", code=LOG_CODE, level="debug"
    )
    inspect(
        message=f"Size of cube = {format_bytes(cube.nbytes)}",
        code=LOG_CODE,
        level="debug",
    )

    # validate input bands

    if "bands" not in cube.dims:
        raise ValueError("cube must have a bands dimension")
    if cube.sizes["bands"] != 2:
        raise ValueError("expected 2 bands: data and mask")

    try:
        data = cube.sel(bands="data", drop=True)
    except KeyError as e:
        raise ValueError("expected a band labeled 'data'") from e

    nan_pct = (data.isnull().sum().item() / data.size) * 100
    inspect(
        message=f"NaNs in data band = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )
        
    try:
        mask_band = cube.sel(bands="mask", drop=True)
    except KeyError as e:
        raise ValueError("expected a band labeled 'mask'") from e

    nan_pct = (mask_band.isnull().sum().item() / mask_band.size) * 100
    inspect(
        message=f"NaNs in mask band = {nan_pct:.2f}%",
        code=LOG_CODE,
        level="debug",
    )

    filled_data = xr.apply_ufunc(
        nearest_neighbour_fill,
        data,
        mask_band,
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )

    return filled_data.expand_dims(bands=["data"])


def apply_metadata(metadata: CollectionMetadata, context: Context) -> CollectionMetadata:
    """
    Returns the expected cube metadata, after applying this UDF, based on input metadata.
    """
    return metadata.filter_bands(["data"])
