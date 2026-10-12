import unittest

import torch

from scratch_lm.config import ModelConfig
from scratch_lm.model import ScratchLanguageModel


class ScratchLanguageModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=256, block_size=16, n_embd=32, n_layer=2, n_head=4, dropout=0.0
        )
        self.model = ScratchLanguageModel(self.config)

    def test_logits_shape_and_loss(self):
        tokens = torch.randint(0, 256, (2, 12))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_rejects_sequence_longer_than_context(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, 256, (1, 17)))

    def test_byte_encoding_round_trip(self):
        text = "Interview answer: hello!"
        self.assertEqual(self.model.decode(self.model.encode(text)), text)


if __name__ == "__main__":
    unittest.main()

