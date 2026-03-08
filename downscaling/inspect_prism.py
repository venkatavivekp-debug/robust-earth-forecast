import zipfile
import os

zip_path = "data_raw/prism/prism_temp.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data_raw/prism")

print("Extracted PRISM files")
