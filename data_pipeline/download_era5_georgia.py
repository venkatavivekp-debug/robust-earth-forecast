from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import cdsapi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 2m temperature for Georgia")
    parser.add_argument("--year", type=int, required=True, help="Year, e.g. 2023")
    parser.add_argument("--month", type=int, required=True, help="Month 1-12")
    parser.add_argument(
        "--out",
        type=str,
        default="data_raw/era5_georgia_temp.nc",
        help="Output NetCDF path",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=[35.0, -85.0, 30.0, -80.0],
        metavar=("N", "W", "S", "E"),
        help="Bounding box [north west south east]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    days_in_month = calendar.monthrange(args.year, args.month)[1]
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved ERA5 file to: {output_path}")


if __name__ == "__main__":
    main()
