from __future__ import annotations

import unittest

import torch

from models.unet_downscaler import UNetDownscaler


class UNetDownscalerSmokeTest(unittest.TestCase):
    def test_forward_shape_and_gradients(self) -> None:
        torch.manual_seed(0)
        model = UNetDownscaler(in_channels=12, out_channels=1, base_channels=8)
        x = torch.randn(2, 3, 4, 9, 11)

        y = model(x, target_size=(31, 39))
        self.assertEqual(tuple(y.shape), (2, 1, 31, 39))

        loss = y.square().mean()
        loss.backward()

        self.assertIsNotNone(model.enc1.net[1].weight.grad)
        self.assertGreater(float(model.enc1.net[1].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.enc2.net[1].weight.grad)
        self.assertGreater(float(model.enc2.net[1].weight.grad.abs().sum()), 0.0)

    def test_decoder_receives_skip_concatenations(self) -> None:
        model = UNetDownscaler(in_channels=12, out_channels=1, base_channels=8)
        seen_channels: dict[str, int] = {}

        def capture(name: str):
            def hook(_module, inputs, _output) -> None:
                seen_channels[name] = int(inputs[0].shape[1])

            return hook

        h2 = model.dec2.register_forward_hook(capture("dec2"))
        h1 = model.dec1.register_forward_hook(capture("dec1"))
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 4, 8, 10), target_size=(32, 40))
        finally:
            h2.remove()
            h1.remove()

        self.assertEqual(seen_channels["dec2"], 8 * 4 + 8 * 2)
        self.assertEqual(seen_channels["dec1"], 8 * 2 + 8)


if __name__ == "__main__":
    unittest.main()
