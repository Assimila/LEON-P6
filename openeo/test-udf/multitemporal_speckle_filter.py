import unittest

import numpy as np
import xarray as xr

from udf import multitemporal_speckle_filter


class Test_multitemporal_speckle_filter(unittest.TestCase):
    def setUp(self):
        self.radius = 2
        self.window_size = 5
        self.context = {
            "radius": self.radius,
            "window_size": self.window_size,
        }
        self.t_size = 9
        self.spatial_coords = {"y": range(5), "x": range(5)}
        self.coords = {**self.spatial_coords, "t": range(self.t_size)}
        self.middle_t = self.t_size // 2

    def test_constant_unchanged(self):
        data = np.full((5, 5, self.t_size), 5.0)
        input_xr = xr.DataArray(data, coords=self.coords, dims=["y", "x", "t"])

        output_xr = multitemporal_speckle_filter.apply_datacube(input_xr, self.context)

        expected = np.full((5, 5, self.t_size), np.nan)
        expected[2, 2, 2:7] = 5.0  # only these central pixels remain as not NaN
        expected_xr = xr.DataArray(expected, coords=self.coords, dims=["y", "x", "t"])

        xr.testing.assert_equal(output_xr, expected_xr)

    def test_temporal_outlier_smoothed(self):
        data = np.full((5, 5, self.t_size), 5.0)
        data[2, 2, self.middle_t] = 50.0
        input_xr = xr.DataArray(data, coords=self.coords, dims=["y", "x", "t"])

        output_xr = multitemporal_speckle_filter.apply_datacube(input_xr, self.context)
        filtered = output_xr.sel(y=2, x=2, t=self.middle_t).item()

        # spatial mean at spike time: 1 pixel at 50 + 12 neighbours at 5 (radius=2 disk)
        mu = (50.0 + 12 * 5.0) / 13
        ni_spike = 50.0 / mu
        sum_ni = ni_spike + 4 * 1.0  # four other times in window_size=5 temporal window
        expected = mu / self.window_size * sum_ni
        self.assertAlmostEqual(filtered, expected)

    def test_extra_dims(self):
        band0 = np.full((5, 5, self.t_size), 5.0)
        band1 = np.full((5, 5, self.t_size), 5.0)
        band1[2, 2, self.middle_t] = 50.0
        input_xr = xr.DataArray(
            np.stack([band0, band1]),
            coords={"bands": [0, 1], **self.coords},
            dims=["bands", "y", "x", "t"],
        )

        output_xr = multitemporal_speckle_filter.apply_datacube(input_xr, self.context)

        for band in (0, 1):
            with self.subTest(band=band):
                band_input = input_xr.sel(bands=band)
                expected = multitemporal_speckle_filter.apply_datacube(
                    band_input, self.context
                )
                xr.testing.assert_equal(output_xr.sel(bands=band), expected)

    def test_radius_not_int(self):
        input_xr = xr.DataArray(
            np.full((5, 5, self.t_size), 5.0),
            coords=self.coords,
            dims=["y", "x", "t"],
        )
        context = {"radius": 2.1, "window_size": self.window_size}

        with self.assertRaises(ValueError):
            multitemporal_speckle_filter.apply_datacube(input_xr, context)

    def test_radius_too_small(self):
        input_xr = xr.DataArray(
            np.full((5, 5, self.t_size), 5.0),
            coords=self.coords,
            dims=["y", "x", "t"],
        )
        context = {"radius": 0, "window_size": self.window_size}

        with self.assertRaises(ValueError):
            multitemporal_speckle_filter.apply_datacube(input_xr, context)

    def test_window_size_not_int(self):
        input_xr = xr.DataArray(
            np.full((5, 5, self.t_size), 5.0),
            coords=self.coords,
            dims=["y", "x", "t"],
        )
        context = {"radius": self.radius, "window_size": 5.1}

        with self.assertRaises(ValueError):
            multitemporal_speckle_filter.apply_datacube(input_xr, context)

    def test_window_size_invalid(self):
        input_xr = xr.DataArray(
            np.full((5, 5, self.t_size), 5.0),
            coords=self.coords,
            dims=["y", "x", "t"],
        )

        for window_size in (2, 4):
            with self.subTest(window_size=window_size):
                context = {"radius": self.radius, "window_size": window_size}
                with self.assertRaises(ValueError):
                    multitemporal_speckle_filter.apply_datacube(input_xr, context)

    def test_insufficient_temporal_data(self):
        input_xr = xr.DataArray(
            np.full((5, 5, 4), 5.0),
            coords={"y": range(5), "x": range(5), "t": range(4)},
            dims=["y", "x", "t"],
        )

        with self.assertRaises(ValueError):
            multitemporal_speckle_filter.apply_datacube(input_xr, self.context)
