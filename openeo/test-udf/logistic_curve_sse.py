import unittest

import numpy as np
import xarray as xr

from udf import logistic_curve_sse


class Test_logistic_curve_sse(unittest.TestCase):
    def setUp(self):
        self.window_size = 11
        self.steepness_parameter = -2.0
        self.context = {
            "window_size": self.window_size,
            "steepness_parameter": self.steepness_parameter,
        }
        self.t_size = 15
        self.coords = {"y": [0], "x": [0], "t": range(self.t_size)}

    def _make_cube(self, series: list[float]) -> xr.DataArray:
        data = np.array(series, dtype=float).reshape(1, 1, self.t_size)
        return xr.DataArray(data, coords=self.coords, dims=["y", "x", "t"])

    def test_constant_zero_sse(self):
        input_xr = self._make_cube([5.0] * self.t_size)

        output_xr = logistic_curve_sse.apply_datacube(input_xr, self.context)

        # windows size 11, so expect 5 NaNs at each end
        expected_t = [np.nan] * 5 + [0.0] * 5 + [np.nan] * 5
        expected = xr.DataArray(
            np.array(expected_t, dtype=float).reshape(1, 1, self.t_size),
            coords=self.coords,
            dims=["y", "x", "t"],
        )
        xr.testing.assert_equal(output_xr, expected)

    def test_nonzero_sse_for_step_change(self):
        input_xr = self._make_cube([10.0] * 7 + [4.5] + [1.0] * 7)

        output_xr = logistic_curve_sse.apply_datacube(input_xr, self.context)
        middle = output_xr.isel(t=7).item()

        self.assertGreater(middle, 0)

    def test_extra_dims(self):
        band0 = [5.0] * self.t_size
        band1 = [10.0] * 7 + [4.5] + [1.0] * 7
        input_xr = xr.DataArray(
            np.stack(
                [
                    np.array(band0, dtype=float).reshape(1, 1, self.t_size),
                    np.array(band1, dtype=float).reshape(1, 1, self.t_size),
                ]
            ),
            coords={"bands": [0, 1], **self.coords},
            dims=["bands", "y", "x", "t"],
        )

        output_xr = logistic_curve_sse.apply_datacube(input_xr, self.context)

        for band in (0, 1):
            with self.subTest(band=band):
                band_input = input_xr.sel(bands=band)
                expected = logistic_curve_sse.apply_datacube(band_input, self.context)
                xr.testing.assert_equal(output_xr.sel(bands=band), expected)

    def test_window_size_not_int(self):
        input_xr = self._make_cube([5.0] * self.t_size)
        context = {
            "window_size": 11.2,
            "steepness_parameter": self.steepness_parameter,
        }

        with self.assertRaises(ValueError):
            logistic_curve_sse.apply_datacube(input_xr, context)

    def test_window_size_invalid(self):
        input_xr = self._make_cube([5.0] * self.t_size)

        for window_size in (2, 4):
            with self.subTest(window_size=window_size):
                context = {
                    "window_size": window_size,
                    "steepness_parameter": self.steepness_parameter,
                }
                with self.assertRaises(ValueError):
                    logistic_curve_sse.apply_datacube(input_xr, context)

    def test_steepness_not_float(self):
        input_xr = self._make_cube([5.0] * self.t_size)
        context = {
            "window_size": self.window_size,
            "steepness_parameter": "-2.0",
        }

        with self.assertRaises(ValueError):
            logistic_curve_sse.apply_datacube(input_xr, context)

    def test_insufficient_temporal_data(self):
        input_xr = xr.DataArray(
            np.full((1, 1, 10), 5.0),
            coords={"y": [0], "x": [0], "t": range(10)},
            dims=["y", "x", "t"],
        )

        with self.assertRaises(ValueError):
            logistic_curve_sse.apply_datacube(input_xr, self.context)
