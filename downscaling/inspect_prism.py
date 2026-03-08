import zipfile
import os

zip_path = "data_raw/prism/prism_tmean_20230101.zip"
extract_path = "data_raw/prism"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("PRISM dataset extracted successfully")
