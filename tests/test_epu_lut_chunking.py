import unittest

import torch

from tools.epu_lut_chunking import bf16_lut_encoded_size, plan_bf16_lut_chunks


class EPULUTChunkingTest(unittest.TestCase):
    def test_1536_channels_split_into_240_channel_luts(self):
        chunks = plan_bf16_lut_chunks(1536)
        self.assertEqual(chunks, [(0, 240), (240, 240), (480, 240),
                                  (720, 240), (960, 240), (1200, 240),
                                  (1440, 96)])
        self.assertTrue(all(bf16_lut_encoded_size(size) < 4096
                            for _, size in chunks))
        self.assertGreater(bf16_lut_encoded_size(1536), 4095)

    def test_pointwise_channel_split_is_exact(self):
        torch.manual_seed(0)
        x = torch.randn(1, 7, 3, 1536, dtype=torch.float32)
        reference = torch.tanh(x) * 0.5 + x
        pieces = [torch.tanh(x[..., start:start + size]) * 0.5
                  + x[..., start:start + size]
                  for start, size in plan_bf16_lut_chunks(x.shape[-1])]
        reconstructed = torch.cat(pieces, dim=-1)
        self.assertEqual(torch.max(torch.abs(reference - reconstructed)).item(),
                         0.0)


if __name__ == "__main__":
    unittest.main()
