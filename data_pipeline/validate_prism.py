from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import rasterio

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".bil", ".asc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PRISM rasters and optionally convert BIL files to GeoTIFF"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to PRISM zip archive, raster file, or raster directory",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Directory for extracted zip contents (used only when --path is a zip)",
    )
    parser.add_argument(
        "--convert-bil-to-geotiff",
        action="store_true",
        help="Convert discovered .bil files to GeoTIFF",
    )
    parser.add_argument(
        "--geotiff-dir",
        type=str,
        default="data_raw/prism/geotiff",
        help="Output directory for converted GeoTIFF files",
    )
    return parser.parse_args()


def collect_raster_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [path]

    if path.is_dir():
        return sorted(
            file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    return []


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def inspect_raster(raster_path: Path) -> None:
    with rasterio.open(raster_path) as src:
        print(
            f"{raster_path} | shape=({src.height}, {src.width}) "
            f"dtype={src.dtypes[0]} crs={src.crs} nodata={src.nodata}"
        )


def convert_bil_to_geotiff(raster_paths: list[Path], output_dir: Path) -> None:
    bil_files = [path for path in raster_paths if path.suffix.lower() == ".bil"]
    if not bil_files:
        print("No .bil files found for conversion.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for bil_path in bil_files:
        out_path = output_dir / f"{bil_path.stem}.tif"
        with rasterio.open(bil_path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff")

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())

        print(f"Converted: {bil_path} -> {out_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = Path(args.extract_dir) if args.extract_dir else input_path.with_suffix("")
        print(f"Extracting zip archive: {input_path} -> {extract_dir}")
        input_path = extract_zip(input_path, extract_dir)

    raster_files = collect_raster_files(input_path)
    if not raster_files:
        raise RuntimeError(
            f"No supported PRISM rasters found at {input_path}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    print(f"Found {len(raster_files)} raster file(s).")
    for raster_file in raster_files:
        inspect_raster(raster_file)

    if args.convert_bil_to_geotiff:
        convert_bil_to_geotiff(raster_files, Path(args.geotiff_dir))


if __name__ == "__main__":
    main()
