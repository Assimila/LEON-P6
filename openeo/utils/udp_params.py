"""
reusable UDP Parameters
"""

import numpy as np
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

TEMPORAL_EXTENT = Parameter.temporal_interval(
    name="temporal_extent",
    description="The temporal extent of the data to process. Pad by 3 months on each end to allow for full logistic curve fit.",
    default=["2019-10-01", "2025-04-01"],
)

CANOPY_COVER_THRESHOLD = Parameter.number(
    name="canopy_cover_threshold",
    description=(
        "Minimum canopy cover to be considered forest. "
        "Units: percent. "
        "Should be one of (10, 20, 30, 40, 50, 60, 70, 80, 90). "
        "A value of 30 implies > 30% canopy cover. "
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

S1_ORBIT_STATE = Parameter.string(
    name="orbit_state",
    description="The orbit state to process (ascending, descending)",
    default="ascending",
)

S1_RELATIVE_ORBIT = Parameter.number(
    name="relative_orbit",
    description="The relative orbit to process",
    default=101,
)

SPECKLE_FILTER_RADIUS = Parameter.number(
    name="speckle_filter_radius",
    description="The radius of the speckle filter. Units: pixels",
    default=2,
)

SPECKLE_FILTER_CV_NOISE = Parameter.number(
    name="speckle_filter_cv_noise",
    description=(
        "The coefficient of variation of the speckle noise. "
        "In multi-looked data, this takes the value of 1 / sqrt(ENL), "
        "where ENL is the Equivalent Number of Looks."
    ),
    default=1.0 / np.sqrt(4),
)

SPECKLE_FILTER_TEMPORAL_WINDOW = Parameter.number(
    name="speckle_filter_temporal_window",
    description="The temporal window of the speckle filter. Units: pixels",
    default=5,
)

SPATIAL_RESOLUTION = Parameter.number(
    name="spatial_resolution",
    description=(
        "The spatial resolution to process. Units: meters. ",
        "Data are resampled to this resolution. ",
    ),
    default=30,
)

LOGISTIC_WINDOW_SIZE = Parameter.number(
    name="logistic_window_size",
    description="The window size of the logistic curve fit. Units: pixels",
    default=11,
)

LOGISTIC_STEEPNESS_PARAMETER = Parameter.number(
    name="logistic_steepness_parameter",
    description="The steepness parameter of the logistic curve fit. Units: dimensionless",
    default=-2.0,
)


TEMPORAL_VARIABILITY_THRESHOLD = Parameter.number(
    name="temporal_variability_threshold",
    description="Exclude any pixels with temporal standard deviation below this threshold. Units: dB",
    default=0.5,
)

FLATTENING_THRESHOLD = Parameter.number(
    name="flattening_threshold",
    description="Exclude any pixels with flattening below this threshold. Units: dimensionless",
    default=0.12,
)

LOGISTIC_SSE_PERCENTILE = Parameter.number(
    name="logistic_sse_percentile",
    description="Exclude any pixels with Sum of Squared Errors (SSE) above this percentile. Units: fraction",
    default=0.95,
)


# outputs from other UDPs

FOREST_BASELINE_DATACUBE = Parameter.datacube(
    name="forest_baseline_datacube",
    description=(
        "The forest baseline datacube to process. ",
        "This is a pixel mask, where 1 = natural forest. "
        "Dimensions: [bands, y, x]. "
        "Bands: [B0]. ",
    ),
)


SENTINEL_1_DATACUBE = Parameter.datacube(
    name="sentinel_1_datacube",
    description=(
        "The Sentinel-1 datacube to process. ",
        "Dimensions: [bands, y, x]. "
        "Bands: [min_sse, min_sse_t, sd, p05, p95]. ",
    ),
)
