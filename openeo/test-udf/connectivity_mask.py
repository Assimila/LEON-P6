import unittest

import numpy as np
import xarray as xr
from udf import connectivity_mask


class Test_connectivity_mask(unittest.TestCase):
    def test_no_excluded_pixels(self):

        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = input_xr.copy()
        expected.values[:] = 0

        for min_connected_area in range(1, 10):
            with self.subTest(min_connected_area=min_connected_area):
                context = {
                    "spatial_resolution": 1.0,
                    "min_connected_area": min_connected_area,
                }

                output_xr = connectivity_mask.apply_datacube(input_xr, context)

                xr.testing.assert_equal(output_xr, expected)

    def test_all_excluded_pixels(self):

        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = input_xr.copy()

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 10,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_diagonal_connectivity(self):

        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = input_xr.copy()
        expected.values[:] = 0

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 5.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_all_zeros(self):

        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = input_xr.copy()
        expected.values[:] = 0

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 1.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_all_ones(self):
        input_array = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = input_xr.copy()
        expected.values[:] = 0

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 1.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_selective_masking(self):
        # 3x3 blob (9 px) kept; isolated 2-px blob masked when min_pixels=3
        input_array = np.array(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = xr.DataArray(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                ],
                dtype=bool,
            ),
            dims=["y", "x"],
        )

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 3.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_spatial_resolution_conversion(self):
        # Matches forest-baseline: 10m spatial_resolution.
        # 3x3 blob = 900 m^2; min 1000 m^2 -> ceil(1000/100)=10 -> mask all 9.
        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected = input_xr.copy()

        context = {
            "spatial_resolution": 10.0,
            "min_connected_area": 1000.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_spatial_resolution_ceil(self):
        # ceil(250 / (10*10)) = 3; 2-px feature masked, 3-px feature kept
        input_array = np.array(
            [
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = xr.DataArray(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                ],
                dtype=bool,
            ),
            dims=["y", "x"],
        )

        context = {
            "spatial_resolution": 10.0,
            "min_connected_area": 250.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_non_bool_float_input(self):
        input_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, np.nan, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        expected = xr.DataArray(
            np.zeros((5, 5), dtype=bool),
            dims=["y", "x"],
        )

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 8.0,
        }

        # Mute openEO inspect() user logger for the expected dtype warning
        with self.assertLogs("openeo.udf.debug.user", level="WARNING"):
            output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_extra_dims(self):
        # Two bands with different feature sizes; vectorize over non-spatial dims
        band0 = np.array(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        band1 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(
            np.stack([band0, band1]),
            dims=["bands", "y", "x"],
        )

        expected = xr.DataArray(
            np.stack(
                [
                    np.zeros((5, 5), dtype=bool),
                    band1,
                ]
            ),
            dims=["bands", "y", "x"],
        )

        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 3.0,
        }

        output_xr = connectivity_mask.apply_datacube(input_xr, context)

        xr.testing.assert_equal(output_xr, expected)

    def test_spatial_resolution_zero(self):

        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        context = {
            "spatial_resolution": 0.0,
            "min_connected_area": 1.0,
        }

        with self.assertRaises(ValueError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_spatial_resolution_negative(self):
        input_xr = xr.DataArray(np.ones((5, 5), dtype=bool), dims=["y", "x"])
        context = {
            "spatial_resolution": -1.0,
            "min_connected_area": 1.0,
        }

        with self.assertRaises(ValueError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_spatial_resolution_wrong_type(self):
        input_xr = xr.DataArray(np.ones((5, 5), dtype=bool), dims=["y", "x"])
        context = {
            "spatial_resolution": "100",
            "min_connected_area": 1.0,
        }

        with self.assertRaises(TypeError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_min_connected_area_negative(self):
        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": -1.0,
        }

        with self.assertRaises(ValueError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_min_connected_area_zero(self):
        # Allowed by < 0 check, then fails when min_pixels = ceil(0 / pixel_area) = 0
        input_xr = xr.DataArray(np.ones((5, 5), dtype=bool), dims=["y", "x"])
        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 0.0,
        }

        with self.assertRaises(ValueError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_min_connected_area_wrong_type(self):
        input_xr = xr.DataArray(np.ones((5, 5), dtype=bool), dims=["y", "x"])
        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": None,
        }

        with self.assertRaises(TypeError):
            connectivity_mask.apply_datacube(input_xr, context)

    def test_min_connected_area_too_large(self):
        input_array = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        context = {
            "spatial_resolution": 1.0,
            "min_connected_area": 25,
        }

        with self.assertRaises(ValueError):
            connectivity_mask.apply_datacube(input_xr, context)
