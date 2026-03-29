from __future__ import annotations

import argparse
import calendar
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 2m temperature for Georgia")
    parser.add_argument("--year", type=int, default=2023, help="Year (default: 2023)")
    parser.add_argument("--month", type=int, default=1, help="Month 1-12 (default: 1)")
    parser.add_argument(
        "--out",
        type=str,
        default="data_raw/era5_georgia_temp.nc",
        help="Output NetCDF path (default: data_raw/era5_georgia_temp.nc)",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=[35.0, -85.0, 30.0, -80.0],
        metavar=("N", "W", "S", "E"),
        help="Bounding box [north west south east] for Georgia",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload even if output file already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 1 <= args.month <= 12:
        raise ValueError("--month must be in 1..12")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"ERA5 file already exists, skipping download: {output_path}")
        print("Use --overwrite to force a new download.")
        return

    cdsapirc_path = Path.home() / ".cdsapirc"
    if not cdsapirc_path.exists():
        raise RuntimeError(
            "ERA5 download requires CDS API credentials. "
            "Create ~/.cdsapirc first: https://cds.climate.copernicus.eu/how-to-api"
        )

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'cdsapi'. Install requirements first: pip install -r requirements.txt"
        ) from exc

    days_in_month = calendar.monthrange(args.year, args.month)[1]

    print("Starting ERA5 download...")
    print(f"  year={args.year} month={args.month:02d}")
    print(f"  area={args.area}")
    print(f"  output={output_path}")

    try:
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["2m_temperature"],
                "year": str(args.year),
                "month": f"{args.month:02d}",
                "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "area": args.area,
                "format": "netcdf",
            },
            str(output_path),
        )
    except Exception as exc:
        raise RuntimeError(f"ERA5 download failed: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ERA5 download finished but output file is missing or empty")

    print(f"Saved ERA5 file to: {output_path}")


if __name__ == "__main__":
    main()
