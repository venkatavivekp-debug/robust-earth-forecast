from __future__ import annotations

import unittest

import torch

from training.train_downscaler import compute_training_loss, spatial_gradient_loss


class TrainingLossTest(unittest.TestCase):
    def test_spatial_gradient_loss_zero_for_matching_fields(self) -> None:
        y = torch.randn(2, 1, 8, 9)
        self.assertAlmostEqual(float(spatial_gradient_loss(y, y)), 0.0, places=7)

    def test_spatial_gradient_loss_detects_spatial_detail(self) -> None:
        pred = torch.zeros(1, 1, 6, 6)
        target = torch.zeros(1, 1, 6, 6)
        target[:, :, 3:, :] = 1.0
        self.assertGreater(float(spatial_gradient_loss(pred, target)), 0.0)

    def test_gradient_mode_increases_loss_when_final_prediction_is_smooth(self) -> None:
        raw = torch.zeros(1, 1, 6, 6)
        loss_target = torch.zeros_like(raw)
        final = torch.zeros_like(raw)
        target = torch.zeros_like(raw)
        target[:, :, 3:, :] = 1.0

        mse_loss, _ = compute_training_loss(
            raw_preds=raw,
            loss_target=loss_target,
            final_preds=final,
            y=target,
            loss_mode="mse",
            l1_weight=0.1,
            grad_weight=0.05,
        )
        grad_loss, _ = compute_training_loss(
            raw_preds=raw,
            loss_target=loss_target,
            final_preds=final,
            y=target,
            loss_mode="mse_grad",
            l1_weight=0.1,
            grad_weight=0.05,
        )
        self.assertGreater(float(grad_loss), float(mse_loss))


if __name__ == "__main__":
    unittest.main()
