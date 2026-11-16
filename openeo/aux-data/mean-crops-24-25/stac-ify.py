"""
Create STAC metadata for local auxiliary data so that openEO can load it.

https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-backend/docs/load_stac.html

Source: Google Earth Engine Dynamic World (temporal mean of crops, 2024-2025).
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
"""

from __future__ import annotations

import datetime
import os
import shutil
import sys
from pathlib import Path

import pystac
import rasterio as rio
from pystac import CatalogType, Collection, Item
from pystac.layout import CustomLayoutStrategy
from rio_stac.stac import create_stac_item

SCRIPT_DIR = Path(__file__).resolve().parent
TILES_DIR = SCRIPT_DIR / "tiles"
STAC_DIR = SCRIPT_DIR / "STAC"
DESCRIPTION_PATH = SCRIPT_DIR / "description.txt"

RASTER_EXT_V2 = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"

COLLECTION_ID = "mean-crops-24-25-uganda-32636"
COLLECTION_TITLE = "Mean crops 2024-25 - Uganda tiles (EPSG:32636)"
COLLECTION_DESCRIPTION = (
    "100 km UTM tiles of Dynamic World temporal mean cropland probability "
    "2024-2025 over Uganda (band crops, 10 m, EPSG:32636)."
)

BAND_NAME = "crops"
BAND_DESCRIPTION = "Mean cropland probability (Dynamic World, 2024-2025)"

ITEM_DATETIME = datetime.datetime(2024, 1, 1)
TEMPORAL_START = datetime.datetime(2024, 1, 1)
TEMPORAL_END = datetime.datetime(2025, 12, 31)


def build_item_for_tile(tif_path: Path, description: str) -> pystac.Item:
    asset_href = f"../../tiles/{tif_path.name}"
    with rio.open(tif_path) as dataset:
        item = create_stac_item(
            source=dataset,
            id=tif_path.stem,
            input_datetime=ITEM_DATETIME,
            asset_roles=["data"],
            asset_href=asset_href,
            with_proj=True,
            with_raster=False,
            with_eo=False,
        )

        band = 0
        data_type = dataset.dtypes[band]
        nodata = dataset.nodata
        scale = dataset.scales[band]
        offset = dataset.offsets[band]

    item.common_metadata.description = description

    if RASTER_EXT_V2 not in item.stac_extensions:
        item.stac_extensions.append(RASTER_EXT_V2)

    bands_obj: dict = {
        "name": BAND_NAME,
        "description": BAND_DESCRIPTION,
        "data_type": data_type,
    }
    if nodata is not None:
        bands_obj["nodata"] = nodata
    (asset,) = item.assets.values()
    asset.extra_fields["bands"] = [bands_obj]
    asset.extra_fields["raster:scale"] = scale
    asset.extra_fields["raster:offset"] = offset

    return item


def stac_collection_href(collection: Collection, parent_dir: str, is_root: bool) -> str:
    """Path for this collection's JSON file when normalizing the STAC tree.

    Writes `collection.json` in the catalog root
    directory instead of a subdirectory named after the collection id.

    Args:
        collection:
        parent_dir:
        is_root: Whether this collection is the normalize/save root

    Returns:
        Absolute path to `collection.json` under `parent_dir`.
    """
    return os.path.join(parent_dir, "collection.json")


def stac_item_href(item: Item, parent_dir: str) -> str:
    """Path for each Item JSON when normalizing the STAC tree.

    Puts every item under an `items/` subfolder next
    to `collection.json`, named `{item.id}.json`.

    Args:
        item: The STAC Item
        parent_dir: Absolute directory of the parent Collection on disk

    Returns:
        Absolute path to `items/<item.id>.json` under `parent_dir`.
    """
    return os.path.join(parent_dir, "items", f"{item.id}.json")


def main() -> None:
    tif_paths = sorted(TILES_DIR.glob("*.tif"))
    if not tif_paths:
        print(f"No GeoTIFFs found in {TILES_DIR}/", file=sys.stderr)
        sys.exit(1)

    description = DESCRIPTION_PATH.read_text()

    if STAC_DIR.exists():
        shutil.rmtree(STAC_DIR)
    STAC_DIR.mkdir(parents=True)

    collection = Collection(
        id=COLLECTION_ID,
        title=COLLECTION_TITLE,
        description=COLLECTION_DESCRIPTION,
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent([[TEMPORAL_START, TEMPORAL_END]]),
        ),
    )

    for i, tif_path in enumerate(tif_paths, start=1):
        print(f"[{i}/{len(tif_paths)}] {tif_path.name} ...")
        item = build_item_for_tile(tif_path, description)
        item.validate()
        collection.add_item(item)

    collection.update_extent_from_items()
    collection.extent.temporal = pystac.TemporalExtent([[TEMPORAL_START, TEMPORAL_END]])

    root_href = os.fspath(STAC_DIR.resolve())
    strategy = CustomLayoutStrategy(
        collection_func=stac_collection_href,
        item_func=stac_item_href,
    )
    collection.normalize_and_save(
        root_href=root_href,
        catalog_type=CatalogType.SELF_CONTAINED,
        strategy=strategy,
    )
    collection.validate()

    print(f"Wrote {len(tif_paths)} items under {STAC_DIR / 'items'}/")
    print(f"Wrote {STAC_DIR / 'collection.json'}")


if __name__ == "__main__":
    main()
