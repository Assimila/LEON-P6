"""
reusable UDP Parameters
"""

import shapely
from openeo.api.process import Parameter

spatial_extent = {
    "west": 30.5503711040000994,
    "south": 1.0709279050000799,
    "east": 31.2229521229999989,
    "north": 1.5469373050000299,
}

# spatial extent as dict of Polygon geometry
spatial_extent = shapely.geometry.mapping(
    shapely.box(
        xmin=spatial_extent["west"],
        ymin=spatial_extent["south"],
        xmax=spatial_extent["east"],
        ymax=spatial_extent["north"],
    )
)

# This is deliberately a `Parameter.geojson`, not a `Parameter.spatial_extent`,
# because CDSE does not implement `reduce_spatial` method, so we have to use `aggregate_spatial` instead.
SPATIAL_EXTENT = Parameter.geojson(name="spatial_extent", default=spatial_extent)


CANOPY_COVER_THRESHOLD = Parameter.number(
    name="canopy_cover_threshold",
    description=(
        "Minimum canopy cover to be considered forest. "
        "Units: percent. "
        "See collection CLMS_TCD_PANTROPICAL_10M_YEARLY_V1. "
        "A value of 10 implies > 10% canopy cover. "
        "A value of 20 implies > 20% canopy cover. "
    ),
    default=30,
)


NATURAL_FOREST_THRESHOLD = Parameter.number(
    name="natural_forest_threshold",
    description=(
        "Minimum likelihood to be considered natural forest. "
        "Units: fraction. "
        "A value of 0.08 implies > 8% likelihood. "
    ),
    default=0.08,
)

MIN_CONNECTED_AREA = Parameter.number(
    name="min_connected_area",
    description="Minimum connected area to be considered forest. Units: m^2",
    default=10000,
)
