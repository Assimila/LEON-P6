import unittest

import numpy as np
import pandas as pd
import xarray as xr

from udf import idxmin_t


def expected_decimal_year(ts: pd.Timestamp) -> float:
    doy = ts.dayofyear
    days = 366 if ts.is_leap_year else 365
    return ts.year + (doy - 1) / days


class Test_idxmin_t(unittest.TestCase):
    def test_uniform_minimum(self):
        t_coords = pd.to_datetime(["2020-01-01", "2020-07-02", "2021-01-01"])
        data = np.array(
            [
                # y0
                [
                    [10.0, 1.0, 10.0],  # x0: t0, t1, t2
                    [10.0, 1.0, 10.0],  # x1
                ],
                # y1
                [
                    [10.0, 1.0, 10.0],
                    [10.0, 1.0, 10.0],
                ],
            ]
        )
        input_xr = xr.DataArray(
            data,
            coords={"y": [0, 1], "x": [0, 1], "t": t_coords},
            dims=["y", "x", "t"],
        )

        min_date = t_coords[1]
        expected = xr.DataArray(
            np.full((2, 2), expected_decimal_year(min_date)),
            coords={"y": [0, 1], "x": [0, 1]},
            dims=["y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)

    def test_spatially_varying_minimum(self):
        t_coords = pd.to_datetime(
            ["2020-01-01", "2020-07-02", "2021-01-01", "2021-07-02"]
        )
        data = np.full((4, 2, 2), 10.0)  # (t, y, x)
        data[0, 0, 0] = 1.0  # (y, x) = (0, 0) min at t0
        data[1, 0, 1] = 1.0  # (y, x) = (0, 1) min at t1
        data[2, 1, 0] = 1.0  # (y, x) = (1, 0) min at t2
        data[3, 1, 1] = 1.0  # (y, x) = (1, 1) min at t3

        input_xr = xr.DataArray(
            data,
            coords={"t": t_coords, "y": [0, 1], "x": [0, 1]},
            dims=["t", "y", "x"],
        )

        expected_values = np.array(
            [
                [expected_decimal_year(t_coords[0]), expected_decimal_year(t_coords[1])],
                [expected_decimal_year(t_coords[2]), expected_decimal_year(t_coords[3])],
            ]
        )
        expected = xr.DataArray(
            expected_values,
            coords={"y": [0, 1], "x": [0, 1]},
            dims=["y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)

    def test_leap_year_conversion(self):
        t_coords = pd.to_datetime(["2020-01-01", "2020-02-29", "2020-07-01"])
        data = np.array([[[10.0, 1.0, 10.0]]])  # (y, x, t)
        input_xr = xr.DataArray(
            data,
            coords={"y": [0], "x": [0], "t": t_coords},
            dims=["y", "x", "t"],
        )

        expected = xr.DataArray(
            np.array([[expected_decimal_year(t_coords[1])]]),
            coords={"y": [0], "x": [0]},
            dims=["y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)

    def test_non_leap_year_conversion(self):
        t_coords = pd.to_datetime(["2021-01-01", "2021-03-01", "2021-07-01"])
        data = np.array([[[10.0, 1.0, 10.0]]])  # (y, x, t)
        input_xr = xr.DataArray(
            data,
            coords={"y": [0], "x": [0], "t": t_coords},
            dims=["y", "x", "t"],
        )

        expected = xr.DataArray(
            np.array([[expected_decimal_year(t_coords[1])]]),
            coords={"y": [0], "x": [0]},
            dims=["y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)

    def test_extra_dims(self):
        t_coords = pd.to_datetime(["2020-01-01", "2020-07-02", "2021-01-01"])
        band0 = np.array(
            [
                [[10.0, 1.0, 10.0], [10.0, 1.0, 10.0]],
                [[10.0, 1.0, 10.0], [10.0, 1.0, 10.0]],
            ]
        )  # (y, x, t)
        band1 = np.array(
            [
                [[1.0, 10.0, 10.0], [1.0, 10.0, 10.0]],
                [[1.0, 10.0, 10.0], [1.0, 10.0, 10.0]],
            ]
        )  # (y, x, t)
        input_xr = xr.DataArray(
            np.stack([band0, band1]),
            coords={"bands": [0, 1], "y": [0, 1], "x": [0, 1], "t": t_coords},
            dims=["bands", "y", "x", "t"],
        )

        expected = xr.DataArray(
            np.stack(
                [
                    np.full((2, 2), expected_decimal_year(t_coords[1])),
                    np.full((2, 2), expected_decimal_year(t_coords[0])),
                ]
            ),
            coords={"bands": [0, 1], "y": [0, 1], "x": [0, 1]},
            dims=["bands", "y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)

    def test_nan_ignored(self):
        t_coords = pd.to_datetime(["2020-01-01", "2020-07-02", "2021-01-01"])
        data = np.array([[[10.0, np.nan, 1.0]]])  # (y, x, t)
        input_xr = xr.DataArray(
            data,
            coords={"y": [0], "x": [0], "t": t_coords},
            dims=["y", "x", "t"],
        )

        expected = xr.DataArray(
            np.array([[expected_decimal_year(t_coords[2])]]),
            coords={"y": [0], "x": [0]},
            dims=["y", "x"],
        )

        output_xr = idxmin_t.apply_datacube(input_xr, {})
        xr.testing.assert_allclose(output_xr, expected)
