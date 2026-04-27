from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
import socket
import zipfile

import requests

DATE_PATTERN = re.compile(r"(19|20)\d{6}")
SUPPORTED_RASTER_EXTENSIONS = {".bil", ".tif", ".tiff", ".asc"}
REQUIRED_EXTRACTED_EXTENSIONS = {".bil", ".tif", ".tiff", ".hdr"}
INVALID_DOWNLOAD_MESSAGE = "Invalid PRISM download — likely blocked or incorrect URL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare PRISM daily temperature data")
    parser.add_argument(
        "--start-date",
        type=str,
        default="20230101",
        help="First date in YYYYMMDD format (default: 20230101)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of consecutive days from --start-date (ignored if --end-date is set)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive end YYYYMMDD; if set with --start-date, downloads every day in the inclusive range (overrides --days)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="tmean",
        choices=["tmean", "tmin", "tmax"],
        help="PRISM variable",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us",
        help="PRISM region for NACSE endpoint (default: us)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="4km",
        help="PRISM resolution for NACSE endpoint (default: 4km)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_raw/prism",
        help="Directory for downloaded and extracted PRISM files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1_000_000,
        help="Minimum zip size in bytes to consider download valid",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload even if raster files for a date already exist",
    )
    return parser.parse_args()


def prism_url(region: str, resolution: str, variable: str, date_str: str) -> str:
    return f"https://services.nacse.org/prism/data/get/{region}/{resolution}/{variable}/{date_str}"


def validate_zip_file(zip_path: Path, min_bytes: int) -> None:
    if not zip_path.exists() or zip_path.stat().st_size < min_bytes:
        raise RuntimeError(INVALID_DOWNLOAD_MESSAGE)

    if not zipfile.is_zipfile(zip_path):
        preview = zip_path.read_bytes()[:256].lower()
        if b"<html" in preview or b"web services" in preview or b"doctype" in preview:
            raise RuntimeError(INVALID_DOWNLOAD_MESSAGE)
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {zip_path}")


def extract_and_cleanup_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)

    extracted_exts = {path.suffix.lower() for path in out_dir.rglob("*") if path.is_file()}
    if not (extracted_exts & REQUIRED_EXTRACTED_EXTENSIONS):
        raise RuntimeError(
            f"Extraction completed but no expected PRISM outputs found (.bil/.tif/.hdr) in {out_dir}"
        )

    zip_path.unlink(missing_ok=True)


def validate_prism_files(path: str | Path) -> list[Path]:
    prism_dir = Path(path)
    if not prism_dir.exists() or not prism_dir.is_dir():
        raise FileNotFoundError(f"PRISM directory not found: {prism_dir}")

    raster_files = sorted(
        file
        for file in prism_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_RASTER_EXTENSIONS
    )
    if not raster_files:
        raise RuntimeError(f"No PRISM raster files found in {prism_dir}")

    missing_dates = [file.name for file in raster_files if not DATE_PATTERN.search(file.name)]
    if missing_dates:
        sample = ", ".join(missing_dates[:3])
        raise RuntimeError(
            "PRISM raster filenames must include YYYYMMDD. "
            f"Invalid filenames found: {sample}"
        )

    return raster_files


def date_already_present(out_dir: Path, date_str: str) -> bool:
    matches = [
        file
        for file in out_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_RASTER_EXTENSIONS
        and DATE_PATTERN.search(file.name)
        and DATE_PATTERN.search(file.name).group(0) == date_str
    ]
    return bool(matches)


def download_single_date(
    *,
    out_dir: Path,
    variable: str,
    region: str,
    resolution: str,
    date_str: str,
    timeout: int,
    min_bytes: int,
) -> None:
    url = prism_url(region=region, resolution=resolution, variable=variable, date_str=date_str)
    zip_path = out_dir / f"prism_{variable}_{region}_{resolution}_{date_str}.zip"

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError(INVALID_DOWNLOAD_MESSAGE)

    with zip_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    validate_zip_file(zip_path, min_bytes=min_bytes)
    extract_and_cleanup_zip(zip_path, out_dir)


def main() -> None:
    args = parse_args()

    try:
        start_date = datetime.strptime(args.start_date, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("--start-date must be in YYYYMMDD format") from exc

    if args.end_date is not None:
        try:
            end_date = datetime.strptime(args.end_date, "%Y%m%d")
        except ValueError as exc:
            raise ValueError("--end-date must be in YYYYMMDD format") from exc
        n_days = (end_date - start_date).days + 1
        if n_days < 1:
            raise ValueError("--end-date must be on or after --start-date")
    else:
        if args.days < 1:
            raise ValueError("--days must be >= 1")
        n_days = int(args.days)

    # Fail fast when network/DNS is unavailable.
    try:
        socket.gethostbyname("services.nacse.org")
    except OSError as exc:
        raise RuntimeError(
            "Cannot resolve services.nacse.org. Check internet/DNS connectivity before downloading PRISM."
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_dates: list[str] = []
    skipped_dates: list[str] = []

    for offset in range(n_days):
        date_value = start_date + timedelta(days=offset)
        date_str = date_value.strftime("%Y%m%d")

        if not args.overwrite and date_already_present(out_dir, date_str):
            skipped_dates.append(date_str)
            continue

        try:
            download_single_date(
                out_dir=out_dir,
                variable=args.variable,
                region=args.region,
                resolution=args.resolution,
                date_str=date_str,
                timeout=args.timeout,
                min_bytes=args.min_bytes,
            )
            downloaded_dates.append(date_str)
            print(f"Downloaded and extracted PRISM for {date_str}")
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed PRISM HTTP request for {date_str}: {exc}") from exc

    raster_files = validate_prism_files(out_dir)

    print(f"PRISM directory ready: {out_dir}")
    if downloaded_dates:
        print(f"Downloaded dates: {', '.join(downloaded_dates)}")
    if skipped_dates:
        print(f"Skipped existing dates: {', '.join(skipped_dates)}")
    print(f"Raster files found: {len(raster_files)}")


if __name__ == "__main__":
    main()
