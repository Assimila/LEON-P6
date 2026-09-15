# openEO workarounds

Here we document workarounds that we had to implement to circumvent bugs or deficiencies in openEO / the CDSE openEO backend.

Each item is a TODO: delete the workaround from our code once the backend supports the native process.

## `load_stac` / job results STAC nodata

It is currently not possible to express to the backend "there is no nodata encoding".

- https://forum.dataspace.copernicus.eu/t/load-stac-configure-nodata/5267

Job results STAC does not encode `dtype` and `nodata`,
so the next `load_stac_from_job` imposes a default nodata value of 0.

- https://forum.dataspace.copernicus.eu/t/round-trip-nodata/5512

Valid zeros become NaN, and "empty" tiles are then optimised away by the backend.
This creates strange "missing tile" artifacts.

- https://forum.dataspace.copernicus.eu/t/apply-neighborhood-skips-tiles/5277

### Workarounds

Natural-forest STAC is written with nodata=255.

Natural-forest `load_stac` opts into scale/offset via an undocumented
feature flag `"apply_raster_scale_and_offset"`.

## UDP parameters ignored in band math

Native comparisons (`>`, `>=`, `<=`) and arithmetic do not apply UDP
`Parameter` values. 
The backend silently fails!

- https://forum.dataspace.copernicus.eu/t/udp-parameter-not-applied/5282

### Workaround

A custom UDF `udf/binary_operator.py` in every UDP that thresholds a parameter.
Used in place of quick native band math.

Inline scripts use native `>` / `>=` / `<=` because they have concrete numbers.

## `not` / invert after `merge_cubes`

`not` is applied as bitwise complement when the cube dtype is not a real
boolean (common after `merge_cubes`). There is no client-side way to
inspect the backend cell type.

- https://forum.dataspace.copernicus.eu/t/invert-not-of-a-pixel-mask/5323

### Workaround

`utils.logical_not()` is `(cube == 0)`.
This seems to work most of the time, but intermediate results must be checked manually.

## Vector cube `filter_bbox` not supported

> OpenEoApiError: [400] ProcessParameterInvalid: The value passed for parameter 'data' in process 'filter_bbox' is invalid: Expected raster cube but got vector cube.

### Workaround

Custom UDF `udf/vectorcube_filter_bbox.py` instead of native `filter_bbox` predefined process.

## Almost no vector-cube processing

After `aggregate_spatial`, it is not possible to do much with the resulting vector cube, such as merging, renaming dimension labels, or running UDFs.

- https://forum.dataspace.copernicus.eu/t/merge-vector-cubes/5425

### Workarounds

Carefully prepare a raster datacube, with band labels set in advance.
After `aggregate_spatial` no further processing is possible.

## Chaining UDPs with intermediate results fails

https://forum.dataspace.copernicus.eu/t/error-chaining-upds/5528

### Workaround

None.
Cannot check intermediate results.

## Vector outputs do not support `filename_prefix`

Vector outputs will overwrite each other.
Only one output per type is returned from `MultiResult`.

- https://forum.dataspace.copernicus.eu/t/multiresult-with-geojson/5416

### Workaround

None.

## CSV is indexed by feature_index only

Cannot access geometry properties (GeoJSON feature properties).

- https://forum.dataspace.copernicus.eu/t/access-to-geojson-properties/5418

### Workaround

Use Parquet instead.

## Band math across different cubes

`cube_a.band(...) & cube_b.band(...)` raises

> BandMathException: 'Band math' between bands of different data cubes is not supported yet.

### Workaround

First, `merge_cubes` into a single datacube, then band-math on the stacked cube. 
