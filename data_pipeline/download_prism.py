from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import zipfile

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download daily PRISM temperature archive")
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date in YYYYMMDD format (example: 20230101)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="tmean",
        choices=["tmean", "tmin", "tmax"],
        help="PRISM temperature variable",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_raw/prism",
        help="Directory where the zip file is saved",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded zip archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    date = datetime.strptime(args.date, "%Y%m%d")
    year = date.strftime("%Y")

    filename = f"PRISM_{args.variable}_stable_4kmD1_{args.date}_bil.zip"
    url = f"https://ftp.prism.oregonstate.edu/daily/{args.variable}/{year}/{filename}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / filename

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with zip_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print(f"Downloaded PRISM archive: {zip_path}")

    if args.extract:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(out_dir)
        print(f"Extracted PRISM archive to: {out_dir}")


if __name__ == "__main__":
    main()
