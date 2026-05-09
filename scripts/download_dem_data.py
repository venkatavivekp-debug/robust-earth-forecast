from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USGS_3DEP_IMAGE_SERVER = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DEFAULT_GEORGIA_BBOX = "-85,30,-80,35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a reproducible DEM source for the Georgia PRISM domain.")
    parser.add_argument("--output-dir", default="data_raw/static/source_dem")
    parser.add_argument("--bbox", default=DEFAULT_GEORGIA_BBOX, help="min_lon,min_lat,max_lon,max_lat in EPSG:4326")
    parser.add_argument(
        "--source",
        choices=["usgs_3dep_image_service"],
        default="usgs_3dep_image_service",
        help="Public DEM source. Current implementation uses the USGS 3DEP Elevation ImageServer.",
    )
    parser.add_argument("--size", default="1200,1200", help="Export image size as width,height pixels")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bbox(value: str) -> Tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must contain min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox order: {value}")
    return min_lon, min_lat, max_lon, max_lat


def parse_size(value: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or min(parts) < 16:
        raise ValueError("--size must contain width,height >= 16")
    return parts[0], parts[1]


def request_usgs_export(args: argparse.Namespace) -> Dict[str, Any]:
    bbox = parse_bbox(args.bbox)
    width, height = parse_size(args.size)
    params = {
        "bbox": ",".join(f"{v:g}" for v in bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "f": "json",
    }
    response = requests.get(USGS_3DEP_IMAGE_SERVER, params=params, timeout=int(args.timeout))
    response.raise_for_status()
    payload = response.json()
    if "href" not in payload:
        raise RuntimeError(f"USGS export did not return a download href: {payload}")
    return {
        "source": args.source,
        "service_url": USGS_3DEP_IMAGE_SERVER,
        "request_url": f"{USGS_3DEP_IMAGE_SERVER}?{urlencode(params)}",
        "bbox": list(bbox),
        "size": [width, height],
        "response": payload,
        "download_url": payload["href"],
    }


def download_file(url: str, output_path: Path, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source != "usgs_3dep_image_service":
        raise ValueError(f"Unsupported DEM source: {args.source}")

    export = request_usgs_export(args)
    bbox_tag = "_".join(str(v).replace("-", "m").replace(".", "p") for v in export["bbox"])
    width, height = export["size"]
    dem_path = output_dir / f"usgs_3dep_georgia_{bbox_tag}_{width}x{height}.tif"
    metadata_path = dem_path.with_suffix(".metadata.json")

    if dem_path.exists() and not args.overwrite:
        raise FileExistsError(f"DEM already exists: {dem_path}. Use --overwrite to replace it.")

    download_file(str(export["download_url"]), dem_path, timeout=int(args.timeout))
    metadata = {
        **export,
        "downloaded_at_unix": int(time.time()),
        "output_path": str(dem_path),
        "notes": (
            "USGS 3DEP Elevation ImageServer export. This is a real DEM-derived raster, "
            "requested over the ERA5/PRISM Georgia bbox in EPSG:4326. The GeoTIFF remains ignored by git."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote DEM: {dem_path}")
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
