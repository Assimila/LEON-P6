import unittest

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box
from udf import vectorcube_filter_bbox

BBOX_4326 = {
    "west": 8.0,
    "south": 50.0,
    "east": 8.15,
    "north": 50.15,
}


class Test_vectorcube_filter_bbox(unittest.TestCase):

    def test_happy_path(self):
        # define in WGS84, then project to UTM 32N.
        # Keep clear of the bbox edges: reprojection only moves corner vertices,
        # so parallels/meridians become straight chords and edges that coincide
        # in WGS84 diverge into slivers in UTM.

        outside = box(9.0, 51.0, 9.1, 51.1)
        inside = box(8.02, 50.02, 8.08, 50.08)
        partial = box(8.05, 50.05, 8.2, 50.2)

        feature_index = [0, 1, 2]

        geometries = gpd.GeoDataFrame(
            {"some-property": [10, 11, 12]},
            geometry=[outside, inside, partial],
            crs="EPSG:4326",
            index=feature_index,
        ).to_crs("EPSG:32632")

        cube = xr.DataArray(
            np.array([10.0, 20.0, 30.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )

        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertTrue(out_geoms.geometry.iloc[0].equals(geometries.geometry.iloc[1]))
        self.assertEqual(out_cube.sizes["geometry"], 1)
        np.testing.assert_array_equal(out_cube.values, [20.0])

    def test_same_crs_no_reproject(self):
        outside = box(9.0, 51.0, 9.1, 51.1)
        inside = box(8.02, 50.02, 8.08, 50.08)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            {"some-property": [10, 11]},
            geometry=[outside, inside],
            crs="EPSG:4326",
            index=feature_index,
        )
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertTrue(out_geoms.geometry.iloc[0].equals(inside))
        np.testing.assert_array_equal(out_cube.values, [20.0])

    def test_explicit_bbox_crs(self):
        # Geoms in UTM; bbox given in the same UTM CRS via spatial_extent["crs"].
        inside = box(8.02, 50.02, 8.08, 50.08)
        outside = box(9.0, 51.0, 9.1, 51.1)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            {"some-property": [10, 11]},
            geometry=[outside, inside],
            crs="EPSG:4326",
            index=feature_index,
        ).to_crs("EPSG:32632")

        bbox_utm = box(8.0, 50.0, 8.15, 50.15)
        bbox_utm = (
            gpd.GeoSeries([bbox_utm], crs="EPSG:4326").to_crs("EPSG:32632").iloc[0]
        )
        minx, miny, maxx, maxy = bbox_utm.bounds

        context = {
            "spatial_extent": {
                "west": minx,
                "south": miny,
                "east": maxx,
                "north": maxy,
                "crs": "EPSG:32632",
            }
        }
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertTrue(out_geoms.geometry.iloc[0].equals(geometries.geometry.iloc[1]))
        np.testing.assert_array_equal(out_cube.values, [20.0])

    def test_geometries_dimension_name(self):
        inside = box(8.02, 50.02, 8.08, 50.08)
        outside = box(9.0, 51.0, 9.1, 51.1)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            geometry=[outside, inside],
            crs="EPSG:4326",
            index=feature_index,
        )
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometries"],
            coords={"geometries": feature_index},
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertEqual(out_cube.sizes["geometries"], 1)
        np.testing.assert_array_equal(out_cube.values, [20.0])

    def test_missing_geometry_dimension_raises(self):
        geometries = gpd.GeoDataFrame(
            geometry=[box(8.02, 50.02, 8.08, 50.08)],
            crs="EPSG:4326",
        )
        cube = xr.DataArray(np.array([1.0, 2.0]), dims=["time"])
        context = {"spatial_extent": BBOX_4326}

        with self.assertRaises(ValueError):
            vectorcube_filter_bbox.apply_vectorcube(geometries, cube, context)

    def test_keeps_none(self):
        outside_a = box(9.0, 51.0, 9.1, 51.1)
        outside_b = box(7.0, 49.0, 7.1, 49.1)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            geometry=[outside_a, outside_b],
            crs="EPSG:4326",
            index=feature_index,
        )
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 0)
        self.assertEqual(out_cube.sizes["geometry"], 0)

    def test_keeps_all(self):
        a = box(8.02, 50.02, 8.08, 50.08)
        b = box(8.09, 50.09, 8.12, 50.12)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            {"some-property": [10, 11]},
            geometry=[a, b],
            crs="EPSG:4326",
            index=feature_index,
        )
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 2)
        self.assertEqual(out_cube.sizes["geometry"], 2)
        np.testing.assert_array_equal(out_cube.values, [10.0, 20.0])

    def test_extra_dims(self):
        outside = box(9.0, 51.0, 9.1, 51.1)
        inside = box(8.02, 50.02, 8.08, 50.08)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            geometry=[outside, inside],
            crs="EPSG:4326",
            index=feature_index,
        )
        cube = xr.DataArray(
            np.array([[[10.0, 10.0], [10.0, 10.0]], [[20.0, 20.0], [20.0, 20.0]]]),
            dims=["geometry", "time", "bands"],
            coords={
                "geometry": feature_index,
                "time": ["t0", "t1"],
                "bands": ["B1", "B2"],
            },
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertEqual(out_cube.dims, ("geometry", "time", "bands"))
        self.assertEqual(out_cube.sizes["geometry"], 1)
        self.assertEqual(out_cube.sizes["time"], 2)
        self.assertEqual(out_cube.sizes["bands"], 2)
        np.testing.assert_array_equal(out_cube.values, [[[20.0, 20.0], [20.0, 20.0]]])

    def test_geometries_crs_none_defaults_4326(self):
        outside = box(9.0, 51.0, 9.1, 51.1)
        inside = box(8.02, 50.02, 8.08, 50.08)
        feature_index = [0, 1]

        geometries = gpd.GeoDataFrame(
            geometry=[outside, inside],
            crs=None,
            index=feature_index,
        )
        self.assertIsNone(geometries.crs)
        cube = xr.DataArray(
            np.array([10.0, 20.0]),
            dims=["geometry"],
            coords={"geometry": feature_index},
        )
        context = {"spatial_extent": BBOX_4326}

        out_geoms, out_cube = vectorcube_filter_bbox.apply_vectorcube(
            geometries, cube, context
        )

        self.assertEqual(len(out_geoms), 1)
        self.assertTrue(out_geoms.geometry.iloc[0].equals(inside))
        np.testing.assert_array_equal(out_cube.values, [20.0])
