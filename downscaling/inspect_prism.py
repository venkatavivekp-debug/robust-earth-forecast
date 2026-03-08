import zipfile

zip_path = "data_raw/prism/prism_tmean.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data_raw/prism")

print("PRISM files extracted")
