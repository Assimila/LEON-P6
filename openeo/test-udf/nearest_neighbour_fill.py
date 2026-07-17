import unittest

import numpy as np
import xarray as xr

from udf import nearest_neighbour_fill


def make_cube(
    data: np.ndarray,
    mask: np.ndarray,
) -> xr.DataArray:
    return xr.DataArray(
        np.stack([data, mask]),
        dims=["bands", "y", "x"],
        coords={"bands": ["data", "mask"]},
    )


class Test_nearest_neighbour_fill(unittest.TestCase):
    def test_no_mask_unchanged(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.zeros((2, 2))
        input_xr = make_cube(data, mask)
        expected = xr.DataArray(
            np.stack([data]),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_single_masked_pixel(self):
        data = np.array([[5.0, 0.0]])
        mask = np.array([[0.0, 1.0]])
        input_xr = make_cube(data, mask)
        expected = xr.DataArray(
            np.array([[[5.0, 5.0]]]),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_surrounded_masked_pixel(self):
        data = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        mask = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        input_xr = make_cube(data, mask)
        expected = xr.DataArray(
            np.ones((1, 3, 3)),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_all_masked_one_candidate(self):
        data = np.array(
            [
                [0.0, 0.0],
                [0.0, 7.0],
            ]
        )
        mask = np.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        )
        input_xr = make_cube(data, mask)
        expected = xr.DataArray(
            np.full((1, 2, 2), 7.0),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)

    def test_fill_closest_of_multiple_candidates(self):
        data = np.array([[5.0, np.nan, np.nan, 0.0, np.nan, 10.0]])
        mask = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        input_xr = make_cube(data, mask)
        expected = xr.DataArray(
            np.array([[[5.0, np.nan, np.nan, 10.0, np.nan, 10.0]]]),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)

    def test_no_candidates_raises(self):
        data = np.zeros((2, 2))
        mask = np.ones((2, 2))
        input_xr = make_cube(data, mask)

        with self.assertRaisesRegex(
            ValueError, "no finite valued pixels available for replacement"
        ):
            nearest_neighbour_fill.apply_datacube(input_xr, {})

    def test_wrong_band_count_raises(self):
        input_xr = xr.DataArray(
            np.ones((1, 2, 2)),
            dims=["bands", "y", "x"],
            coords={"bands": ["data"]},
        )

        with self.assertRaisesRegex(ValueError, "expected 2 bands"):
            nearest_neighbour_fill.apply_datacube(input_xr, {})

    def test_missing_bands_dimension_raises(self):
        input_xr = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])

        with self.assertRaisesRegex(ValueError, "bands dimension"):
            nearest_neighbour_fill.apply_datacube(input_xr, {})

    def test_missing_data_band_label_raises(self):
        data = np.array([[5.0, 0.0]])
        mask = np.array([[0.0, 1.0]])
        input_xr = xr.DataArray(
            np.stack([data, mask]),
            dims=["bands", "y", "x"],
            coords={"bands": ["whoops", "mask"]},
        )

        with self.assertRaisesRegex(ValueError, "labeled 'data'"):
            nearest_neighbour_fill.apply_datacube(input_xr, {})

    def test_missing_mask_band_label_raises(self):
        data = np.array([[5.0, 0.0]])
        mask = np.array([[0.0, 1.0]])
        input_xr = xr.DataArray(
            np.stack([data, mask]),
            dims=["bands", "y", "x"],
            coords={"bands": ["data", "whoops"]},
        )

        with self.assertRaisesRegex(ValueError, "labeled 'mask'"):
            nearest_neighbour_fill.apply_datacube(input_xr, {})

    def test_invalid_ndim(self):
        with self.assertRaisesRegex(ValueError, "2 dimensions only"):
            nearest_neighbour_fill.nearest_neighbour_fill(
                np.array([1.0, 0.0]), np.array([0.0, 1.0])
            )

    def test_non_masked_nan_unchanged(self):
        data = np.array(
            [
                [1.0, np.nan],
                [0.0, 1.0],
            ]
        )
        mask = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        output = nearest_neighbour_fill.nearest_neighbour_fill(data, mask)
        expected = np.array(
            [
                [1.0, np.nan],
                [1.0, 1.0],
            ]
        )

        np.testing.assert_array_equal(output, expected)

    def test_extra_dims(self):
        data0 = np.array([[1.0, 0.0], [3.0, 1.0]])
        mask0 = np.array([[0.0, 1.0], [0.0, 0.0]])
        data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask1 = np.zeros((2, 2))
        input_xr = xr.DataArray(
            np.stack(
                [
                    np.stack([data0, data1]),
                    np.stack([mask0, mask1]),
                ]
            ),
            dims=["bands", "t", "y", "x"],
            coords={"t": [0, 1], "bands": ["data", "mask"]},
        )
        expected = xr.DataArray(
            np.array(
                [
                    [
                        [[1.0, 1.0], [3.0, 1.0]],
                        data1,
                    ]
                ]
            ),
            dims=["bands", "t", "y", "x"],
            coords={"t": [0, 1], "bands": ["data"]},
        )

        output_xr = nearest_neighbour_fill.apply_datacube(input_xr, {})

        xr.testing.assert_equal(output_xr, expected)
