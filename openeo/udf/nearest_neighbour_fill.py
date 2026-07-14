"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

from typing import TypedDict

import numpy as np
import scipy.ndimage
from openeo.udf import inspect
import xarray as xr


LOG_CODE = "nearest_neighbour_fill"


class Context(TypedDict):
    # the value which indicates a pixel that should be filled
    sentinel: int | float


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def nearest_neighbour_fill(
    input_array: np.ndarray, sentinel: int | float
) -> np.ndarray:
    """
    Arguments:
        input_array: numpy array. should have 2 dimensions only.
        sentinel: the value which indicates a pixel that should be filled
    """
    if input_array.ndim != 2:
        raise ValueError("input_array should have 2 dimensions only")

    # look for values of `sentinel` in `input_array`
    if np.isnan(sentinel):
        # special case because NaN != NaN
        sentinel_mask = np.isnan(input_array)
    else:
        sentinel_mask = input_array == sentinel

    if not sentinel_mask.any():
        # early return
        return input_array

    # get a mask of finite valued pixels that can be used for replacement
    candidate_mask = np.isfinite(input_array) & ~sentinel_mask

    if not candidate_mask.any():
        raise ValueError("no finite valued pixels available for replacement")

    # scipy.ndimage.distance_transform_edt computes, for each True pixel, the
    # Euclidean distance to the nearest False pixel.

    # Pass ~candidate_mask to create a lookup from every non-candidate pixel (True)
    # to the nearest candidate pixel (False).
    _, nearest_coords = scipy.ndimage.distance_transform_edt(
        ~candidate_mask, return_indices=True
    )

    # nearest_coords is a tuple of coordinate arrays, one per axis.
    # Each array has the same shape as input_array.
    # For pixel (i, j), the index of the nearest candidate is (II[i, j], JJ[i, j]).
    II, JJ = nearest_coords

    output = input_array.copy()
    # For each sentinel pixel, copy the value from its nearest candidate.
    output[sentinel_mask] = input_array[II[sentinel_mask], JJ[sentinel_mask]]
    return output


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Look for values of `sentinel` in `cube`.
    Replace these with the value of the spatially nearest finite valued pixel.
    If multiple pixels are equally close, one of these is chosen.

    If there are no occurances of `sentinel`, return the input cube unchanged (noop).

    If there are occurances of `sentinel`, but no finite valued pixels in the cube,
    raise an error.

    Arguments:
        cube: xarray Dataset
            Should have spatial dimensions "x" and "y".
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

    sentinel = context["sentinel"]
    if not isinstance(sentinel, (int, float)):
        raise TypeError("sentinel has unsupported type")

    output = xr.apply_ufunc(
        nearest_neighbour_fill,
        cube,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        kwargs=dict(sentinel=sentinel),
    )

    return output
