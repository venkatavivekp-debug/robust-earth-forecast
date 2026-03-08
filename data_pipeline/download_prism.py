import requests
import os

url = "https://ftp.prism.oregonstate.edu/daily/tmean/2023/PRISM_tmean_stable_4kmD1_20230101_bil.zip"

os.makedirs("data_raw/prism", exist_ok=True)

out_path = "data_raw/prism/prism_temp.zip"

r = requests.get(url, stream=True)

with open(out_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Downloaded PRISM file:", out_path)
