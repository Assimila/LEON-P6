# Natural forest 2020 — Uganda GEE export

Batch export of [Nature Trace natural forest 2020](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection) (band `B0`, 10 m) for Uganda.

Tiled on a 100 x 100 km UTM grid, for later ingestion into openEO.

## bounding box

Uganda's approximate bounding box in [W, S, E, N] order is 
[29.5734, -1.4823, 35.0003, 4.2341]

Export CRS: [EPSG:32636](https://epsg.io/32636) (UTM zone 36N).

### UTM tiling scheme

- Tile size: 100 km × 100 km (100,000 m)
- Grid alignment: SW corner of each tile on multiples of 100 km in UTM metres
- Grid origin (col=0, row=0 SW): easting 100,000 m, northing −200,000 m
- Grid extent: 7 columns × 7 rows. 49 tiles
- Pixel size in export: 10 m

Tile bounds for column `col` and row `row` (0-based):

```
xmin = 100_000 + col * 100_000
xmax = xmin + 100_000
ymin = -200_000 + row * 100_000
ymax = ymin + 100_000
```

File names use the SW corner in kilometres:
`natural_forest_2020_e{Ekm}_n{Nkm}_32636`.
The `n` suffix keeps its sign (`n-200` for negative km).

Grid extent in WGS84:

- Lower-left corner of the grid (SW of tile 0,0): lon 29.40590, lat −1.80588 degrees
- Upper-right corner of the grid (NE of tile 6,6): lon 35.70338, lat 4.51852 degrees

## Expected products

- One COG GeoTIFF per tile, band `B0`, 10 m, EPSG:32636.
- Values 0–250 (uint8 probabilities)

# Usage

First, ensure that you have Earth Engine credentials configured:

```bash
earthengine authenticate
```

Run the script to export the tiles from Earth Engine to GCS:

```bash
python export_uganda_tiles.py \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_BUCKET
```

## Copy from GCS:

Use the [Google Cloud CLI](https://cloud.google.com/sdk/docs) with credentials that can read the bucket (e.g. `gcloud auth login`), not necessarily the same flow as Earth Engine.

Sync the whole prefix:

```bash
gcloud storage rsync gs://YOUR_BUCKET/LEON-P6/natural-forests-2020/ ./tiles -r
```

## STAC

Generate a STAC Collection and one Item per tile

```bash
python stac-ify.py
```

Outputs:

- `STAC/collection.json` — entry point for openEO `load_stac`
- `STAC/items/*.json` — per-tile items (assets under `../../tiles/`)
