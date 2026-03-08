import requests
import os

url = "https://prism.nacse.org/data/public/4km/tmean/2023/PRISM_tmean_stable_4kmD1_20230101_bil.zip"

os.makedirs("data_raw/prism", exist_ok=True)

r = requests.get(url)

with open("data_raw/prism/prism_temp.zip", "wb") as f:
    f.write(r.content)

print("Downloaded PRISM temperature")
