"""
Create STAC metadata for local auxiliary data so that openEO can load it.

https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-backend/docs/load_stac.html

This dataset comes from Google Earth Engine Dynamic World
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
"""

import datetime

from rio_stac.stac import create_stac_item

data_file = "meancrops_24-25-float32.tif"

item = create_stac_item(
    source=data_file,
    input_datetime=datetime.datetime(2024, 1, 1),
    asset_roles=["data"],
    with_proj=True,
    with_raster=True,
    with_eo=True,
)

item.common_metadata.description = open("description.txt").read()

(asset,) = item.assets.values()

if asset.ext.eo.bands is None:
    raise ValueError("expected 1 band")

(band,) = asset.ext.eo.bands

band.name = "crops"
band.description = "Mean cropland probability (Dynamic World, 2024-2025)"

item.validate()

item.save_object(dest_href="item.json")
