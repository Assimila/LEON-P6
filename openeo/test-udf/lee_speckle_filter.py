import unittest

import numpy as np
import xarray as xr

from udf import lee_speckle_filter


class Test_lee_speckle_filter(unittest.TestCase):
    def setUp(self):
        self.radius = 2
        self.cv_noise = 0.5  # Equivalent Number of Looks (ENL) = 4
        self.context = {"radius": self.radius, "cv_noise": self.cv_noise}
        self.coords = {"y": range(5), "x": range(5)}

    def test_homogeneous_unchanged(self):
        data = np.full((5, 5), 5.0)
        input_xr = xr.DataArray(data, coords=self.coords, dims=["y", "x"])

        output_xr = lee_speckle_filter.apply_datacube(input_xr, self.context)
        # check the centre pixel is unchanged
        self.assertAlmostEqual(output_xr.sel(y=2, x=2).item(), 5.0)

    def test_heterogeneous_smoothing(self):
        data = np.full((5, 5), 1.0)
        data[2, 2] = 10.0
        input_xr = xr.DataArray(data, coords=self.coords, dims=["y", "x"])

        output_xr = lee_speckle_filter.apply_datacube(input_xr, self.context)
        center = output_xr.sel(y=2, x=2).item()
        # check the centre pixel is smoothed
        pixels = [1.0] * 12 + [10.0]  # centre pixel + 12 surrounding pixels due to circular kernel
        mu = np.mean(pixels)
        sigma2 = np.var(pixels)
        W = 1 - self.cv_noise**2 * mu**2 / sigma2
        expected = mu + W * (10.0 - mu)
        self.assertAlmostEqual(center, expected)

    def test_extra_dims(self):
        band0 = np.full((5, 5), 5.0)
        band1 = np.full((5, 5), 1.0)
        band1[2, 2] = 10.0
        input_xr = xr.DataArray(
            np.stack([band0, band1]),
            coords={"bands": [0, 1], **self.coords},
            dims=["bands", "y", "x"],
        )

        output_xr = lee_speckle_filter.apply_datacube(input_xr, self.context)

        for band in (0, 1):
            with self.subTest(band=band):
                # run just this band through the filter
                band_input = input_xr.sel(bands=band)
                expected = lee_speckle_filter.apply_datacube(band_input, self.context)
                # check the output is the same as the expected
                xr.testing.assert_equal(output_xr.sel(bands=band), expected)

    def test_boundary_nans(self):
        """
        Check that the boundary pixels are NaN-filled,
        due to the size of the kernel.
        The boundary should be eroded by the filter.
        """
        data = np.full((5, 5), 5.0)
        input_xr = xr.DataArray(data, coords=self.coords, dims=["y", "x"])

        output_xr = lee_speckle_filter.apply_datacube(input_xr, self.context)

        expected_values = np.full((5, 5), np.nan)
        expected_values[2, 2] = 5.0
        expected = xr.DataArray(expected_values, coords=self.coords, dims=["y", "x"])

        xr.testing.assert_equal(output_xr, expected)

    def test_radius_not_int(self):
        input_xr = xr.DataArray(np.full((5, 5), 5.0), coords=self.coords, dims=["y", "x"])
        context = {"radius": 2.1, "cv_noise": self.cv_noise}

        with self.assertRaises(ValueError):
            lee_speckle_filter.apply_datacube(input_xr, context)

    def test_radius_too_small(self):
        input_xr = xr.DataArray(np.full((5, 5), 5.0), coords=self.coords, dims=["y", "x"])
        context = {"radius": 0, "cv_noise": self.cv_noise}

        with self.assertRaises(ValueError):
            lee_speckle_filter.apply_datacube(input_xr, context)

    def test_cv_noise_not_float(self):
        input_xr = xr.DataArray(np.full((5, 5), 5.0), coords=self.coords, dims=["y", "x"])
        context = {"radius": self.radius, "cv_noise": "0.5"}

        with self.assertRaises(ValueError):
            lee_speckle_filter.apply_datacube(input_xr, context)

    def test_cv_noise_not_positive(self):
        input_xr = xr.DataArray(np.full((5, 5), 5.0), coords=self.coords, dims=["y", "x"])

        for cv_noise in (0.0, -0.5):
            with self.subTest(cv_noise=cv_noise):
                context = {"radius": self.radius, "cv_noise": cv_noise}
                with self.assertRaises(ValueError):
                    lee_speckle_filter.apply_datacube(input_xr, context)
