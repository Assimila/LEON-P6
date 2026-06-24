"""
Create STAC metadata for local auxiliary data so that openEO can load it.

https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-backend/docs/load_stac.html

This dataset comes from Google Earth Engine
https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection
"""

import datetime

from rio_stac.stac import create_stac_item

data_file = "Natural_forest_2020_bugom_32636.tif"

item = create_stac_item(
    source=data_file,
    input_datetime=datetime.datetime(2020, 1, 1),
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

band.name = "B0"
band.description = "Natural forest probabilities (scaled to [0-250])."

item.validate()

item.save_object(dest_href="item.json")
