# Boundary Context Fix

Professor Hu's boundary concern is valid for the current data geometry. The existing ERA5 download used the same bounding box as the PRISM Georgia domain. That means PRISM pixels near the edge are reconstructed from ERA5 fields that have no real atmospheric neighbors outside the clipped box.

## Change Made

`data_pipeline/download_era5_georgia.py` now has:

```bash
--era5-buffer-deg 0.25
```

The default expands the requested ERA5 area by about one ERA5 grid cell on all sides:

- north increases by `0.25`;
- west decreases by `0.25`;
- south decreases by `0.25`;
- east increases by `0.25`.

This does not redownload data automatically. It only makes the next ERA5 acquisition use a buffered request.

## What This Should Help

Buffered ERA5 gives boundary convolution kernels real atmospheric context outside the PRISM target domain. That should reduce the artificial edge condition where the encoder sees clipped atmospheric fields at the same location where the target grid ends.

Expected effects:

- cleaner boundary feature maps;
- less border/interior RMSE separation if missing context is important;
- better edge-gradient behavior near the clipped Georgia box.

## What Is Not Tested Yet

The current local ERA5 files were not redownloaded. The existing dataset loader clips PRISM rasters to the ERA5 bounds, so a buffered ERA5 file also requires careful target-domain handling: the PRISM target should stay on the original Georgia/PRISM domain while ERA5 input keeps the buffer through the encoder.

That target/input-domain separation is not implemented in this pass because it requires a fresh buffered ERA5 file and a controlled dataset-alignment check. For now, the repository has the reproducible download flag and a documented next step.

## Next Check

After redownloading ERA5 with the buffer:

1. confirm ERA5 input tensors include the buffered grid;
2. confirm PRISM targets remain on the intended Georgia domain;
3. rerun border/interior RMSE and boundary-gradient diagnostics;
4. compare against the unbuffered runs using the same model, split, and target mode.
