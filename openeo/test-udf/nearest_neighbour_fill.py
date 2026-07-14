import unittest

import numpy as np
import xarray as xr

from udf import nearest_neighbour_fill

SENTINEL = -999.0


class Test_nearest_neighbour_fill(unittest.TestCase):
    def test_no_sentinels_unchanged(self):
        input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected = input_xr.copy()

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_single_sentinel(self):
        input_array = np.array([[5.0, SENTINEL]])
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected = xr.DataArray(np.array([[5.0, 5.0]]), dims=["y", "x"])

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_surrounded_sentinel(self):
        input_array = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, SENTINEL, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected = xr.DataArray(np.ones((3, 3)), dims=["y", "x"])

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_all_sentinels_one_candidate(self):
        input_array = np.array(
            [
                [SENTINEL, SENTINEL],
                [SENTINEL, 7.0],
            ]
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected = xr.DataArray(np.full((2, 2), 7.0), dims=["y", "x"])

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_closest_of_multiple_candidates(self):

        input_array = np.array(
            [
                [5.0, np.nan, np.nan, SENTINEL, np.nan, 10.0],
            ]
        )
        input_xr = xr.DataArray(input_array, dims=["y", "x"])
        expected_arr = np.array(
            [
                [5.0, np.nan, np.nan, 10.0, np.nan, 10.0],
            ]
        )
        expected = xr.DataArray(expected_arr, dims=["y", "x"])

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)

    def test_no_candidates_raises(self):
        input_array = np.full((2, 2), SENTINEL)
        input_xr = xr.DataArray(input_array, dims=["y", "x"])

        with self.assertRaisesRegex(
            ValueError, "no finite valued pixels available for replacement"
        ):
            nearest_neighbour_fill.apply_datacube(input_xr, {"sentinel": SENTINEL})

    def test_sentinel_wrong_type(self):
        input_xr = xr.DataArray(np.array([[1.0, SENTINEL]]), dims=["y", "x"])

        for sentinel in ("0", None):
            with self.subTest(sentinel=sentinel):
                with self.assertRaises(TypeError):
                    nearest_neighbour_fill.apply_datacube(
                        input_xr, {"sentinel": sentinel}
                    )

    def test_invalid_ndim(self):
        with self.assertRaisesRegex(ValueError, "2 dimensions only"):
            nearest_neighbour_fill.nearest_neighbour_fill(
                np.array([1.0, SENTINEL]), SENTINEL
            )

    def test_nan_sentinel(self):
        input_array = np.array(
            [
                [1.0, np.nan],
                [np.nan, 1.0],
            ]
        )
        output = nearest_neighbour_fill.nearest_neighbour_fill(input_array, np.nan)
        expected = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        np.testing.assert_array_equal(output, expected)

    def test_non_sentinel_nan_unchanged(self):
        input_array = np.array(
            [
                [1.0, np.nan],
                [SENTINEL, 1.0],
            ]
        )
        output = nearest_neighbour_fill.nearest_neighbour_fill(input_array, SENTINEL)
        expected = np.array(
            [
                [1.0, np.nan],
                [1.0, 1.0],
            ]
        )

        np.testing.assert_array_equal(output, expected)

    def test_extra_dims(self):
        slice0 = np.array(
            [
                [1.0, SENTINEL],
                [3.0, 1.0],
            ]
        )
        slice1 = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        input_xr = xr.DataArray(
            np.stack([slice0, slice1]),
            dims=["t", "y", "x"],
        )
        expected = xr.DataArray(
            np.stack(
                [
                    np.array([[1.0, 1.0], [3.0, 1.0]]),
                    slice1,
                ]
            ),
            dims=["t", "y", "x"],
        )

        output_xr = nearest_neighbour_fill.apply_datacube(
            input_xr, {"sentinel": SENTINEL}
        )

        xr.testing.assert_equal(output_xr, expected)
