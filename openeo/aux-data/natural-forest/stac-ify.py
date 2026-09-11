"""
Create STAC metadata for local auxiliary data so that openEO can load it.

https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-backend/docs/load_stac.html

This dataset comes from Google Earth Engine
https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection
"""

import datetime

import rasterio as rio
from rio_stac.stac import create_stac_item, get_raster_info

RASTER_EXT_V2 = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"

data_file = "Natural_forest_2020_bugom_32636.tif"

OVERRIDE_DATA_TYPE = None
OVERRIDE_NODATA = 255
OVERRIDE_SCALE = 0.004  # 1/250: probability stored as 0–250 in uint8
OVERRIDE_OFFSET = None

item = create_stac_item(
    source=data_file,
    input_datetime=datetime.datetime(2020, 1, 1),
    asset_roles=["data"],
    with_proj=True,
    with_raster=False,
    with_eo=False,
)

item.common_metadata.description = open("description.txt").read()
item.stac_extensions.append(RASTER_EXT_V2)
(asset,) = item.assets.values()

with rio.open(data_file) as dataset:
    # use rio_stac to generate raster extension fields
    raster_info = get_raster_info(dataset)
    band_info = raster_info[0]
    data_type = band_info["data_type"]
    nodata = band_info.get("nodata", None)
    scale = band_info["scale"]
    offset = band_info["offset"]

if OVERRIDE_DATA_TYPE is not None:
    data_type = OVERRIDE_DATA_TYPE
if OVERRIDE_NODATA is not None:
    nodata = OVERRIDE_NODATA
if OVERRIDE_SCALE is not None:
    scale = OVERRIDE_SCALE
if OVERRIDE_OFFSET is not None:
    offset = OVERRIDE_OFFSET

# STAC v1.1 field, not yet supported natively by pystac
bands_obj = {
    "name": "B0",
    "description": "Natural forest probabilities (scaled to [0-250]).",
    "data_type": data_type,
}
if nodata is not None:
    # the correct way to indicate "there is no nodata encoding" is to omit the field
    bands_obj["nodata"] = nodata
asset.extra_fields["bands"] = [bands_obj]
# openEO does not read these 2 fields from "bands"
asset.extra_fields["raster:scale"] = scale
asset.extra_fields["raster:offset"] = offset

item.validate()
item.save_object(dest_href="item.json")
