import rasterio

file_path = "data_raw/prism/prism_tmean.tif"

with rasterio.open(file_path) as src:
    data = src.read(1)

    print("Shape:", data.shape)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
