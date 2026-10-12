import json
import tempfile
import unittest
from pathlib import Path

from scratch_lm.data_gate import DataGateError, split_corpus, validate_corpus


class DataGateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.corpus = self.root / "corpus.txt"
        self.manifest = self.root / "corpus.manifest.json"
        self.corpus.write_text("A useful text sample. " * 80, encoding="utf-8")
        self.valid_manifest = {
            "dataset_name": "test corpus",
            "source": "original test fixture",
            "version": "1",
            "license": "CC0-1.0",
            "permitted_uses": ["model_training"],
            "rights_reviewed_by_owner": True,
            "personal_data": "none",
            "split_plan": {
                "method": "sequential_bytes",
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
        }
        self.write_manifest()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.valid_manifest), encoding="utf-8")

    def test_accepts_complete_manifest_and_splits_all_bytes(self):
        raw, manifest = validate_corpus(self.corpus, self.manifest, block_size=8)
        train, validation, test = split_corpus(raw, manifest["split_plan"])
        self.assertEqual(len(train) + len(validation) + len(test), len(raw))
        self.assertGreater(len(train), len(validation))

    def test_rejects_missing_manifest(self):
        self.manifest.unlink()
        with self.assertRaisesRegex(DataGateError, "missing"):
            validate_corpus(self.corpus, self.manifest, block_size=8)

    def test_rejects_unconfirmed_rights(self):
        self.valid_manifest["rights_reviewed_by_owner"] = False
        self.write_manifest()
        with self.assertRaisesRegex(DataGateError, "rights review"):
            validate_corpus(self.corpus, self.manifest, block_size=8)

    def test_rejects_training_use_not_allowed(self):
        self.valid_manifest["permitted_uses"] = ["research"]
        self.write_manifest()
        with self.assertRaisesRegex(DataGateError, "model_training"):
            validate_corpus(self.corpus, self.manifest, block_size=8)

    def test_rejects_unreplaced_template_values(self):
        self.valid_manifest["license"] = "REPLACE_WITH_LICENSE_IDENTIFIER"
        self.write_manifest()
        with self.assertRaisesRegex(DataGateError, "Replace manifest placeholders"):
            validate_corpus(self.corpus, self.manifest, block_size=8)

    def test_rejects_split_fractions_that_do_not_sum_to_one(self):
        self.valid_manifest["split_plan"]["test"] = 0.2
        self.write_manifest()
        with self.assertRaisesRegex(DataGateError, "add up to 1"):
            validate_corpus(self.corpus, self.manifest, block_size=8)

    def test_requires_consent_and_retention_if_personal_data_is_kept(self):
        self.valid_manifest["personal_data"] = "consented"
        self.write_manifest()
        with self.assertRaisesRegex(DataGateError, "consent_or_legal_basis"):
            validate_corpus(self.corpus, self.manifest, block_size=8)


if __name__ == "__main__":
    unittest.main()
