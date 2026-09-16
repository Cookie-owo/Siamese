import unittest
from pathlib import Path

from config import load_config
from models import AttentionResNet34
from preprocessing import attention_branch_preprocess


ROOT = Path(__file__).resolve().parents[1]


class PackageApiTests(unittest.TestCase):
    def test_public_package_identity(self):
        config = load_config(ROOT / "configs" / "release_supplementary_inference.yaml")
        self.assertEqual(config["project"]["algorithm_name"], "Siamese")

    def test_public_model_api(self):
        model = AttentionResNet34(num_class=1)
        self.assertIsNotNone(model)
        self.assertTrue(callable(attention_branch_preprocess))


if __name__ == "__main__":
    unittest.main()
