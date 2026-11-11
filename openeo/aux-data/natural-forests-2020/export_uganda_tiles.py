"""
Export Nature Trace natural forests 2020 (band B0) for Uganda.

https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection

Requires Earth Engine credentials (`earthengine authenticate`).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import TypeAlias

import ee

CRS = "EPSG:32636"
ORIGIN_E_M = 100_000
ORIGIN_N_M = -200_000
TILE_SIZE_M = 100_000
NCOLS = 7
NROWS = 7

DEFAULT_GCS_PREFIX = "LEON-P6/natural-forests-2020"
NATURAL_FOREST_COLLECTION = (
    "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
)
EXPORT_SCALE_M = 10
MAX_PIXELS = int(1e13)
POLL_INTERVAL_S = 30

Row: TypeAlias = int  # Grid row index, northward, 0-based.
Col: TypeAlias = int  # Grid column index, eastward, 0-based.
GridIndex: TypeAlias = tuple[Col, Row]  # (column, row);


@dataclass(frozen=True)
class Tile:
    col: Col
    row: Row
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def name(self) -> str:
        e_km = int(self.xmin / 1000)
        n_km = int(self.ymin / 1000)
        return f"natural_forest_2020_e{e_km}_n{n_km}_32636"

    @property
    def file_name(self) -> str:
        return f"{self.name}.tif"

    def object_key(self, prefix: str) -> str:
        prefix_norm = prefix.strip('/')
        return f"{prefix_norm}/{self.file_name}"

    def gs_uri(self, bucket: str, prefix: str) -> str:
        return f"gs://{bucket}/{self.object_key(prefix)}"

    def geometry(self) -> ee.Geometry:
        return ee.Geometry.Rectangle(
            [self.xmin, self.ymin, self.xmax, self.ymax],
            proj=CRS,
            geodesic=False,
        )


def _tile_at(
    col: Col,
    row: Row,
    *,
    origin_e: float,
    origin_n: float,
    tile_m: float,
) -> Tile:
    xmin = origin_e + col * tile_m
    ymin = origin_n + row * tile_m
    return Tile(
        col=col,
        row=row,
        xmin=xmin,
        ymin=ymin,
        xmax=xmin + tile_m,
        ymax=ymin + tile_m,
    )


def build_all_tiles(
    origin_e: float = ORIGIN_E_M,
    origin_n: float = ORIGIN_N_M,
    tile_m: float = TILE_SIZE_M,
    ncols: int = NCOLS,
    nrows: int = NROWS,
) -> list[Tile]:
    return [
        _tile_at(col, row, origin_e=origin_e, origin_n=origin_n, tile_m=tile_m)
        for row in range(nrows)
        for col in range(ncols)
    ]


def build_only_tiles(
    cells: list[GridIndex],
    *,
    origin_e: float = ORIGIN_E_M,
    origin_n: float = ORIGIN_N_M,
    tile_m: float = TILE_SIZE_M,
) -> list[Tile]:
    tiles: list[Tile] = []
    for col, row in cells:
        tiles.append(
            _tile_at(col, row, origin_e=origin_e, origin_n=origin_n, tile_m=tile_m)
        )
    return tiles


def print_tile_list(
    tiles: list[Tile],
    prefix: str,
    bucket: str | None,
) -> None:
    header = (
        "col\trow\txmin\tymin\txmax\tymax\tname\t"
        + ("gs_uri" if bucket else "object_key")
    )
    print(header)
    for tile in tiles:
        loc = tile.gs_uri(bucket, prefix) if bucket else tile.object_key(prefix)
        print(
            f"{tile.col}\t{tile.row}\t{tile.xmin:.0f}\t{tile.ymin:.0f}\t"
            f"{tile.xmax:.0f}\t{tile.ymax:.0f}\t{tile.name}\t{loc}"
        )


def load_image() -> ee.Image:
    return (
        ee.ImageCollection(NATURAL_FOREST_COLLECTION).mosaic().select("B0")
    )


def submit_exports(
    tiles: list[Tile],
    *,
    project: str,
    bucket: str,
    prefix: str,
) -> list[ee.batch.Task]:
    # uses Earth Engine credentials
    ee.Initialize(project=project)
    
    image = load_image()

    tasks: list[ee.batch.Task] = []
    for tile in tiles:
        region = tile.geometry()
        export_image = image.clip(region)
        task = ee.batch.Export.image.toCloudStorage(
            image=export_image,
            description=tile.name,
            bucket=bucket,
            fileNamePrefix=tile.object_key(prefix).removesuffix(".tif"),
            scale=EXPORT_SCALE_M,
            region=region,
            crs=CRS,
            maxPixels=MAX_PIXELS,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        tasks.append(task)
        print(
            f"Started export {tile.col},{tile.row} task={task.id} -> "
            f"{tile.gs_uri(bucket, prefix)}"
        )
    return tasks


def wait_for_tasks(
    tasks: list[ee.batch.Task],
    poll_interval_s: float = POLL_INTERVAL_S,
) -> None:
    pending = tasks
    failures: list[str] = []

    while pending:
        still: list[ee.batch.Task] = []
        for task in pending:
            status = task.status()
            state = status.get("state")
            label = status.get("description", task.id)
            if state == "COMPLETED":
                print(f"COMPLETED {label} ({task.id})")
                continue
            if state in ("FAILED", "CANCELLED"):
                error = status.get("error_message", status)
                failures.append(f"{label} ({task.id}): {state} — {error}")
                continue
            still.append(task)
        if failures:
            break
        if still:
            states = {task.status().get("state") for task in still}
            print(
                f"Waiting {poll_interval_s:.0f}s — {len(still)} task(s) in {states or 'UNKNOWN'}"
            )
            time.sleep(poll_interval_s)
        pending = still

    if failures:
        print("\nExport failures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


def print_gcloud_sync_hint(bucket: str, prefix: str, dest: str = "./tiles") -> None:
    """Print a `gcloud storage rsync` command for objects under the export prefix."""
    prefix_norm = prefix.strip("/")
    uri = f"gs://{bucket}/{prefix_norm}/"
    print("GCS sync command:")
    print(f"gcloud storage rsync {uri} {dest} -r")


def parse_tile_arg(value: str) -> GridIndex:
    """Parse a `--tiles` value into a grid index.

    Args:
        value: `COL,ROW` string (column first, then row).

    Returns:
        `(Col, Row)` as a `GridIndex`.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected COL,ROW but got {value!r}")
    try:
        col: Col = int(parts[0])
        row: Row = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integers in {value!r}") from exc
    return col, row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Uganda natural forest 2020 tiles (7x7 x 100 km) to GCS."
    )
    parser.add_argument("--project", help="GCP project ID for ee.Initialize")
    parser.add_argument("--bucket", help="GCS bucket name for Earth Engine export")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_GCS_PREFIX,
        help=f"GCS object key prefix (default: {DEFAULT_GCS_PREFIX})",
    )
    parser.add_argument(
        "--list-tiles",
        action="store_true",
        help="Print tile grid and object keys; do not call Earth Engine",
    )
    parser.add_argument(
        "--tiles",
        action="append",
        type=parse_tile_arg,
        metavar="COL,ROW",
        help="Export only these cells as COL,ROW (column east, row north; repeatable)",
    )
    parser.add_argument(
        "--origin-e",
        type=float,
        default=ORIGIN_E_M,
        help=f"Grid origin easting in metres (default {ORIGIN_E_M})",
    )
    parser.add_argument(
        "--origin-n",
        type=float,
        default=ORIGIN_N_M,
        help=f"Grid origin northing in metres (default {ORIGIN_N_M})",
    )
    parser.add_argument(
        "--tile-m",
        type=float,
        default=TILE_SIZE_M,
        help=f"Tile width/height in metres (default {TILE_SIZE_M})",
    )
    parser.add_argument("--ncols", type=int, default=NCOLS)
    parser.add_argument("--nrows", type=int, default=NROWS)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help="Seconds between Earth Engine task status checks",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.tiles:
        tiles = build_only_tiles(
            args.tiles,
            origin_e=args.origin_e,
            origin_n=args.origin_n,
            tile_m=args.tile_m,
        )
    else:
        tiles = build_all_tiles(
            origin_e=args.origin_e,
            origin_n=args.origin_n,
            tile_m=args.tile_m,
            ncols=args.ncols,
            nrows=args.nrows,
        )

    if args.list_tiles:
        print_tile_list(tiles, args.prefix, args.bucket)
        return

    if not args.project:
        parser.error("--project is required unless --list-tiles is set")
    if not args.bucket:
        parser.error("--bucket is required unless --list-tiles is set")

    tasks = submit_exports(
        tiles,
        project=args.project,
        bucket=args.bucket,
        prefix=args.prefix,
    )
    wait_for_tasks(tasks, poll_interval_s=args.poll_interval)
    print_gcloud_sync_hint(args.bucket, args.prefix)


if __name__ == "__main__":
    main()
