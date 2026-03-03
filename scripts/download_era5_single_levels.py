import os
import cdsapi

OUT_DIR = "data_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# Southeast US bounding box: North, West, South, East
AREA = [37.0, -90.0, 28.0, -75.0]

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": "2024",
        "month": "01",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": AREA,
        "format": "netcdf",
    },
    os.path.join(OUT_DIR, "era5_seus_2024_01.nc"),
)

print("Download complete.")
