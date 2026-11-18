import unittest

import numpy as np
import xarray as xr
from udf import pixel_area_ha


class Test_pixel_area_ha(unittest.TestCase):
    def setUp(self):
        self.input_xr = xr.DataArray(
            np.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )

    def test_30m(self):
        expected = xr.DataArray(
            np.array(
                [
                    [0.09, 0.0, 0.09],
                    [0.0, 0.09, 0.0],
                    [0.09, 0.09, 0.0],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        context = {"spatial_resolution": 30.0}
        output_xr = pixel_area_ha.apply_datacube(self.input_xr, context)
        xr.testing.assert_allclose(output_xr, expected)

    def test_10m(self):
        expected = xr.DataArray(
            np.array(
                [
                    [0.01, 0.0, 0.01],
                    [0.0, 0.01, 0.0],
                    [0.01, 0.01, 0.0],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        context = {"spatial_resolution": 10.0}
        output_xr = pixel_area_ha.apply_datacube(self.input_xr, context)
        xr.testing.assert_allclose(output_xr, expected)

    def test_spatial_resolution_int_and_float(self):
        expected = xr.DataArray(
            np.array(
                [
                    [0.09, 0.0, 0.09],
                    [0.0, 0.09, 0.0],
                    [0.09, 0.09, 0.0],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        for spatial_resolution in (30, 30.0):
            with self.subTest(spatial_resolution=spatial_resolution):
                context = {"spatial_resolution": spatial_resolution}
                output_xr = pixel_area_ha.apply_datacube(self.input_xr, context)
                xr.testing.assert_allclose(output_xr, expected)

    def test_nans_preserved(self):
        input_xr = xr.DataArray(
            np.array(
                [
                    [1.0, np.nan],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        expected = xr.DataArray(
            np.array(
                [
                    [0.09, np.nan],
                    [0.0, 0.09],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        context = {"spatial_resolution": 30.0}
        output_xr = pixel_area_ha.apply_datacube(input_xr, context)
        xr.testing.assert_allclose(output_xr, expected)

    def test_bool_mask(self):
        input_xr = xr.DataArray(
            np.array(
                [
                    [True, False],
                    [False, True],
                ],
                dtype=bool,
            ),
            dims=["y", "x"],
        )
        expected = xr.DataArray(
            np.array(
                [
                    [0.09, 0.0],
                    [0.0, 0.09],
                ],
                dtype=float,
            ),
            dims=["y", "x"],
        )
        context = {"spatial_resolution": 30.0}
        output_xr = pixel_area_ha.apply_datacube(input_xr, context)
        xr.testing.assert_allclose(output_xr, expected)

    def test_spatial_resolution_zero(self):
        context = {"spatial_resolution": 0.0}
        with self.assertRaises(ValueError):
            pixel_area_ha.apply_datacube(self.input_xr, context)

    def test_spatial_resolution_negative(self):
        context = {"spatial_resolution": -1.0}
        with self.assertRaises(ValueError):
            pixel_area_ha.apply_datacube(self.input_xr, context)

    def test_spatial_resolution_wrong_type(self):
        for spatial_resolution in ("30", None):
            with self.subTest(spatial_resolution=spatial_resolution):
                context = {"spatial_resolution": spatial_resolution}
                with self.assertRaises(TypeError):
                    pixel_area_ha.apply_datacube(self.input_xr, context)
