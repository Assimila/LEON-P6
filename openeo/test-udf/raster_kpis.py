import unittest

import numpy as np
import xarray as xr
from openeo.metadata import Band, BandDimension, CollectionMetadata, SpatialDimension
from udf import raster_kpis


def make_cube(
    year: np.ndarray,
    forest: np.ndarray,
    cropland: np.ndarray,
) -> xr.DataArray:
    return xr.DataArray(
        np.stack([year, forest, cropland]),
        dims=["bands", "y", "x"],
        coords={
            "bands": [
                "year_of_deforestation",
                "forest_baseline",
                "cropland",
            ]
        },
    )


class Test_raster_kpis(unittest.TestCase):
    def setUp(self):
        self.context = {
            "years": [2020, 2021],
            "spatial_resolution": 30.0,
        }
        # 30 m pixel = 0.09 ha
        self.ha = 0.09

    def test_known_pixels_and_ha_scaling(self):
        year = np.array(
            [
                [2020.5, 2021.0],
                [2019.9, 2022.0],
            ],
            dtype=float,
        )
        forest = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        cropland = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        input_xr = make_cube(year, forest, cropland)

        output_xr = raster_kpis.apply_datacube(input_xr, self.context)

        expected_names = [
            "forest_stock_baseline_ha",
            "deforestation_2020_ha",
            "forest_loss_to_cropland_2020_ha",
            "deforestation_2021_ha",
            "forest_loss_to_cropland_2021_ha",
        ]
        self.assertEqual(list(output_xr.coords["bands"].values), expected_names)

        np.testing.assert_allclose(
            output_xr.sel(bands="forest_stock_baseline_ha").values,
            np.array([[self.ha, self.ha], [0.0, self.ha]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2020_ha").values,
            np.array([[self.ha, 0.0], [0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2021_ha").values,
            np.array([[0.0, self.ha], [0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="forest_loss_to_cropland_2020_ha").values,
            np.array([[self.ha, 0.0], [0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="forest_loss_to_cropland_2021_ha").values,
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

    def test_spatial_resolution_10m(self):
        year = np.array([[2020.0]], dtype=float)
        forest = np.array([[1.0]], dtype=float)
        cropland = np.array([[1.0]], dtype=float)
        input_xr = make_cube(year, forest, cropland)
        context = {"years": [2020], "spatial_resolution": 10.0}

        output_xr = raster_kpis.apply_datacube(input_xr, context)

        np.testing.assert_allclose(
            output_xr.sel(bands="forest_stock_baseline_ha").values,
            np.array([[0.01]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2020_ha").values,
            np.array([[0.01]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="forest_loss_to_cropland_2020_ha").values,
            np.array([[0.01]]),
        )

    def test_spatial_resolution_int_and_float(self):
        year = np.array([[2020.0]], dtype=float)
        forest = np.array([[1.0]], dtype=float)
        cropland = np.array([[0.0]], dtype=float)
        input_xr = make_cube(year, forest, cropland)
        for spatial_resolution in (30, 30.0):
            with self.subTest(spatial_resolution=spatial_resolution):
                context = {
                    "years": [2020],
                    "spatial_resolution": spatial_resolution,
                }
                output_xr = raster_kpis.apply_datacube(input_xr, context)
                np.testing.assert_allclose(
                    output_xr.sel(bands="forest_stock_baseline_ha").values,
                    np.array([[0.09]]),
                )

    def test_mask_nans_behave_as_zero(self):
        # Forest baseline is packaged with year of deforestation, so a NaN forest
        # pixel is nodata and has no year. Cropland is a separate raster, so it
        # can be NaN on a pixel that still has a deforestation year.
        year = np.array([[np.nan, 2020.0]], dtype=float)
        forest = np.array([[np.nan, 1.0]], dtype=float)
        cropland = np.array([[1.0, np.nan]], dtype=float)
        input_xr = make_cube(year, forest, cropland)

        output_xr = raster_kpis.apply_datacube(input_xr, self.context)

        np.testing.assert_allclose(
            output_xr.sel(bands="forest_stock_baseline_ha").values,
            np.array([[0.0, self.ha]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2020_ha").values,
            np.array([[0.0, self.ha]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="forest_loss_to_cropland_2020_ha").values,
            np.array([[0.0, 0.0]]),
        )

    def test_nan_year_matches_no_year_band(self):
        year = np.array([[np.nan]], dtype=float)
        forest = np.array([[1.0]], dtype=float)
        cropland = np.array([[1.0]], dtype=float)
        input_xr = make_cube(year, forest, cropland)

        output_xr = raster_kpis.apply_datacube(input_xr, self.context)

        np.testing.assert_allclose(
            output_xr.sel(bands="forest_stock_baseline_ha").values,
            np.array([[self.ha]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2020_ha").values,
            np.array([[0.0]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="deforestation_2021_ha").values,
            np.array([[0.0]]),
        )
        np.testing.assert_allclose(
            output_xr.sel(bands="forest_loss_to_cropland_2020_ha").values,
            np.array([[0.0]]),
        )

    def test_missing_band_label(self):
        input_xr = xr.DataArray(
            np.ones((2, 1, 1)),
            dims=["bands", "y", "x"],
            coords={"bands": ["year_of_deforestation", "forest_baseline"]},
        )
        with self.assertRaises(KeyError):
            raster_kpis.apply_datacube(input_xr, self.context)

    def test_no_bands_dimension(self):
        input_xr = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])
        with self.assertRaises(ValueError):
            raster_kpis.apply_datacube(input_xr, self.context)

    def test_years_empty(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        context = {"years": [], "spatial_resolution": 30.0}
        with self.assertRaises(ValueError):
            raster_kpis.apply_datacube(input_xr, context)

    def test_years_duplicate(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        context = {"years": [2020, 2020], "spatial_resolution": 30.0}
        with self.assertRaises(ValueError):
            raster_kpis.apply_datacube(input_xr, context)

    def test_years_wrong_type(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        for years in ("2020", None, [2020.5], 2020):
            with self.subTest(years=years):
                context = {"years": years, "spatial_resolution": 30.0}
                with self.assertRaises((TypeError, ValueError)):
                    raster_kpis.apply_datacube(input_xr, context)

    def test_spatial_resolution_zero(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        context = {"years": [2020], "spatial_resolution": 0.0}
        with self.assertRaises(ValueError):
            raster_kpis.apply_datacube(input_xr, context)

    def test_spatial_resolution_negative(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        context = {"years": [2020], "spatial_resolution": -1.0}
        with self.assertRaises(ValueError):
            raster_kpis.apply_datacube(input_xr, context)

    def test_spatial_resolution_wrong_type(self):
        input_xr = make_cube(
            np.array([[2020.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
        for spatial_resolution in ("30", None):
            with self.subTest(spatial_resolution=spatial_resolution):
                context = {
                    "years": [2020],
                    "spatial_resolution": spatial_resolution,
                }
                with self.assertRaises(TypeError):
                    raster_kpis.apply_datacube(input_xr, context)

    def test_apply_metadata_replaces_band_list(self):
        metadata = CollectionMetadata(
            metadata={},
            dimensions=[
                SpatialDimension(name="x", extent=[0, 1]),
                SpatialDimension(name="y", extent=[0, 1]),
                BandDimension(
                    name="bands",
                    bands=[
                        Band("year_of_deforestation"),
                        Band("forest_baseline"),
                        Band("cropland"),
                    ],
                ),
            ],
        )
        updated = raster_kpis.apply_metadata(metadata, self.context)
        self.assertEqual(
            updated.band_names,
            [
                "forest_stock_baseline_ha",
                "deforestation_2020_ha",
                "forest_loss_to_cropland_2020_ha",
                "deforestation_2021_ha",
                "forest_loss_to_cropland_2021_ha",
            ],
        )
