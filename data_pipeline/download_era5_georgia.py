import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2023",
        "month": "01",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": [35, -85, 30, -80],  # Georgia region
        "format": "netcdf",
    },
    "data_raw/era5_georgia_temp.nc",
)
