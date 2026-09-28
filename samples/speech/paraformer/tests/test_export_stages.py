"""Torch export stage contracts, exercised separately with real FunASR installed."""

import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "funasr"))


@unittest.skipUnless(AVAILABLE, "Torch and FunASR export dependencies required")
class ExportStages(unittest.TestCase):
    def test_predictor_matches_upstream_cnn_and_tail_without_cif(self):
        import torch
        from funasr.models.paraformer.cif_predictor import (
            CifPredictorV2,
            CifPredictorV2Export,
        )
        from samples.speech.paraformer.conversion.torch_stages import PredictorStage

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            original = CifPredictorV2(
                idim=512, threshold=1.0, l_order=1, r_order=1, tail_threshold=0.45
            ).eval()
            source = CifPredictorV2Export(original).eval()
            stage = PredictorStage(source).eval()
            for hidden in (torch.zeros(1, 400, 512), torch.randn(1, 400, 512)):
                with torch.inference_mode():
                    weights, _ = source.forward_cnn(hidden, torch.ones(1, 1, 400))
                    padded, expected, _ = source.tail_process_fn(
                        hidden, weights, mask=torch.ones(1, 400)
                    )
                    actual, values = stage(hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(values, padded, rtol=0, atol=0)
                self.assertEqual(tuple(actual.shape), (1, 401))
                self.assertEqual(actual[0, -1].item(), torch.tensor(0.45).item())

    def test_fixed_mask_keeps_geometry_and_uses_each_count(self):
        import torch
        from samples.speech.paraformer.conversion.torch_stages import fixed_mask

        for count in (0, 1, 17, 100):
            result = fixed_mask(torch.tensor([count], dtype=torch.int32), 100)
            self.assertEqual(tuple(result.shape), (1, 100))
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.sum().item(), count)
            self.assertTrue(bool((result[0, :count] == 1).all()))
            self.assertTrue(bool((result[0, count:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
