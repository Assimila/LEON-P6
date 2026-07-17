"""
This is an openEO UDF (User Defined Function).
https://open-eo.github.io/openeo-python-client/udf.html

It targets the "python" runtime, version 3.11.
https://openeofed.dataspace.copernicus.eu/?discover=0&udf-runtime=Python
"""

import math
from typing import TypedDict

import numpy as np
import scipy
import xarray as xr

from openeo.udf import inspect

LOG_CODE = "connectivity_mask"


class Context(TypedDict):
    # pixel area (units: m^2)
    # I can't find a way to access this information within the UDF
    pixel_area: float

    # min connected area (units: m^2)
    min_connected_area: float


def format_bytes(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def small_region_mask(input_array: np.ndarray, min_pixels: int) -> np.ndarray:
    """
    Use scipy.ndimage.label to detect features (clusters of non-zero valued pixels).

    Arguments:
        input_array: a 2D (spatial) boolean array / mask.
            1 = valid pixel. 0 = ignored pixel.
        min_pixels: once clustered into features, exclude features smaller than min_pixels

    Returns:
        a boolean mask of pixels to exclude, based on a minimum feature size.
        1 = excluded pixel.
    """
    if input_array.dtype != bool:
        raise TypeError("input_array must have dtype bool")
    if min_pixels < 1:
        raise ValueError("min_pixels must be 1 or greater")

    # 8-connectivity (diagonals count)
    structure = np.ones((3, 3))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label
    # labelled_features has same shape as input_array.
    # with clusters of pixels labelled by feature id (integer).
    # feature id 0 = pixels not belonging to a feature, because their value in input_array is 0.
    # the first cluster of pixels has feature id = 1.
    labelled_features, _ = scipy.ndimage.label(input_array, structure=structure)

    # 1D array.
    # label_counts[i] is number of pixels with feature id = i.
    label_counts = np.bincount(labelled_features.ravel())

    # map each feature id to a bool (whether it should be masked).
    # label_masked[i] is whether to mask feature id = i.
    labels_masked = label_counts < min_pixels

    # do not mask feature id = 0
    labels_masked[0] = False

    # numpy Integer array indexing https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    # for each pixel in labelled_features (pixel value = feature id)
    # do a positional lookup from labels_masked,
    # essentially mapping from feature id to the corresponding true/false from labels_masked.
    # mask has same shape as labelled_features, which has same shape as input_array
    mask = labels_masked[labelled_features]

    return mask


def apply_datacube(cube: xr.DataArray, context: Context) -> xr.DataArray:
    """
    Generate a mask based on pixel connectivity.
    In the returned mask, pixels with value = 1 correspond to features smaller than min_pixels,
    which should be excluded.

    Arguments:
        cube: xarray DataArray
            Should have spatial dimensions "x" and "y".
            Should be a mask of dtype bool.
        context: user-provided arguments.
    """
    inspect(
        data=cube.sizes, message="Input cube dimensions", code=LOG_CODE, level="debug"
    )
    inspect(
        message=f"Size of cube = {format_bytes(cube.nbytes)}",
        code=LOG_CODE,
        level="debug",
    )

    if cube.dtype != bool:
        inspect(
            message=f"expected dtype bool, got dtype {cube.dtype}", code=LOG_CODE, level="warning"
        )
        # try to convert from float
        cube = cube.notnull() & (cube != 0)

    # parse and validate context

    pixel_area = context["pixel_area"]
    if not isinstance(pixel_area, float | int):
        raise TypeError("pixel_area should be a float")
    if pixel_area <= 0:
        raise ValueError("pixel_area should be positive")

    min_connected_area = context["min_connected_area"]
    if not isinstance(min_connected_area, float | int):
        raise TypeError("min_connected_area should be a float")
    if min_connected_area < 0:
        raise ValueError("min_connected_area should be positive")

    min_pixels = math.ceil(min_connected_area / pixel_area)
    if min_pixels < 1:
        raise ValueError("min_pixels should be at least 1")
    total_pixels = cube.sizes["y"] * cube.sizes["x"]
    if min_pixels >= total_pixels:
        raise ValueError("min_pixels should be less than or equal to the total number of pixels")

    inspect(
        message=f"Applying connectivity mask with min_pixels = {min_pixels}", code=LOG_CODE, level="debug"
    )

    mask = xr.apply_ufunc(
        small_region_mask,
        cube,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        kwargs=dict(min_pixels=min_pixels),
    )

    returned_dtype = mask.dtype
    inspect(
        message=f"returned dtype = {returned_dtype}", code=LOG_CODE, level="debug"
    )

    return mask
