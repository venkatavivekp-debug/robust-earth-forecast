import os
import cdsapi

OUT_DIR = "data_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# SE US bounding box: North, West, South, East
AREA = [37.0, -90.0, 28.0, -75.0]

# Pressure levels in hPa
LEVELS = ["1000", "850", "700", "500", "300"]

# Variables (pressure-level fields)
VARS = [
    "temperature",                 # t
    "u_component_of_wind",         # u
    "v_component_of_wind",         # v
    "relative_humidity",           # r
    "geopotential",                # z (optional but useful)
]

OUT_PATH = os.path.join(OUT_DIR, "era5_pl_seus_2024_01.nc")

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": VARS,
        "pressure_level": LEVELS,

	years = ["2022","2023","2024"]
 	months = ["01","02","03","04","05","06"]

	"day": [f"{d:02d}" for d in range(1, 8)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": AREA,
        "format": "netcdf",
    },
    OUT_PATH,
)

print("Saved:", OUT_PATH)
