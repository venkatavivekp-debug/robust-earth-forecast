from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from datasets.dataset_paths import paths_for_dataset_version
from datasets.prism_dataset import ERA5_PRISM_Dataset
from training.train_downscaler import compute_input_stats, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_PATH = PROJECT_ROOT / "data_processed/static/georgia_prism_topography.nc"


@unittest.skipUnless(
    (PROJECT_ROOT / "data_raw/medium/era5_georgia_multi.nc").exists()
    and (PROJECT_ROOT / "data_raw/medium/prism").exists()
    and STATIC_PATH.exists(),
    "medium ERA5/PRISM/topography data are not available locally",
)
class NormalizationRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        era5_path, prism_path = paths_for_dataset_version("medium", project_root=PROJECT_ROOT)
        cls.dataset = ERA5_PRISM_Dataset(
            era5_path=str(PROJECT_ROOT / era5_path),
            prism_path=str(PROJECT_ROOT / prism_path),
            history_length=3,
            input_set="core4_topo",
            static_covariate_path=str(STATIC_PATH),
            verbose=False,
        )
        _train_set, _val_set, cls.train_indices, _val_indices = split_dataset(cls.dataset, 0.2, 42)

    def test_input_normalize_denormalize_roundtrip(self) -> None:
        mean, std = compute_input_stats(self.dataset, self.train_indices)
        x, _y = self.dataset[self.train_indices[0]]
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)

        normalized = (x - mean_t) / std_t
        recovered = normalized * std_t + mean_t

        self.assertLess(float(torch.max(torch.abs(recovered - x))), 2e-4)

    def test_prism_target_roundtrip(self) -> None:
        values = []
        for idx in self.train_indices:
            _x, y = self.dataset[int(idx)]
            values.append(y.numpy().astype(np.float64))
        stacked = np.stack(values, axis=0)
        mean = float(np.mean(stacked))
        std = float(max(np.std(stacked), 1e-6))

        _x, y = self.dataset[self.train_indices[0]]
        normalized = (y.numpy() - mean) / std
        recovered = normalized * std + mean

        self.assertLess(float(np.max(np.abs(recovered - y.numpy()))), 1e-6)

    def test_static_topography_channels_roundtrip(self) -> None:
        mean, std = compute_input_stats(self.dataset, self.train_indices)
        x, _y = self.dataset[self.train_indices[0]]
        static = x[:, 4:, :, :]
        mean_t = torch.tensor(mean[4:]).view(1, -1, 1, 1)
        std_t = torch.tensor(std[4:]).view(1, -1, 1, 1)

        normalized = (static - mean_t) / std_t
        recovered = normalized * std_t + mean_t

        self.assertLess(float(torch.max(torch.abs(recovered - static))), 2e-4)

    def test_subset_normalizer_can_be_applied_to_larger_subset(self) -> None:
        first_four = self.train_indices[:4]
        first_eight = self.train_indices[:8]
        mean, std = compute_input_stats(self.dataset, first_four)
        mean_t = torch.tensor(mean).view(1, 1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, 1, -1, 1, 1)

        batch = torch.stack([self.dataset[int(idx)][0] for idx in first_eight], dim=0)
        normalized = (batch - mean_t) / std_t
        recovered = normalized * std_t + mean_t

        self.assertLess(float(torch.max(torch.abs(recovered - batch))), 2e-4)


if __name__ == "__main__":
    unittest.main()
