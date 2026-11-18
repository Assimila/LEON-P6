from urllib.parse import urljoin

# TODO: update this to permanent location
NATURAL_FOREST_STAC = "https://s3.waw3-2.cloudferro.com/swift/v1/leon-p6/natural-forests-2020/STAC/collection.json"

# TODO: update this to permanent location
MEAN_CROPS_STAC = "https://s3.waw3-2.cloudferro.com/swift/v1/leon-p6/mean-crops-24-25/STAC/collection.json"

# TODO: point to branch: main
GIT_REPO = "Assimila/LEON-P6"
GIT_BRANCH = "openeo"

GIT_BASE = urljoin(
    "https://raw.githubusercontent.com/",
    f"{GIT_REPO}/refs/heads/{GIT_BRANCH}/",
)

FOREST_BASELINE_UDP = urljoin(GIT_BASE, "openeo/forest-baseline/udp.json")
S1_PROCESSING_UDP = urljoin(GIT_BASE, "openeo/s1-processing/udp.json")
DEFORESTATION_UDP = urljoin(GIT_BASE, "openeo/deforestation/udp.json")
KPIS_UDP = urljoin(GIT_BASE, "openeo/kpis/udp.json")
END_TO_END_UDP = urljoin(GIT_BASE, "openeo/end-to-end/udp.json")
