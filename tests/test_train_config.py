import unittest
from argparse import Namespace

from scripts.train import apply_config_defaults


class TrainConfigTests(unittest.TestCase):
    def test_legacy_config_values_are_applied(self):
        args = Namespace(
            config="configs/reference_experiment_legacy.yaml",
            epochs=100,
            batch_size=32,
            lr=1e-4,
            pos_weight=2.0,
            k=3.0,
            seed=42,
            deterministic=False,
            no_augment=False,
        )
        result = apply_config_defaults(args)
        self.assertEqual(result.epochs, 100)
        self.assertEqual(result.batch_size, 32)
        self.assertEqual(result.lr, 1e-4)
        self.assertEqual(result.pos_weight, 2.0)
        self.assertEqual(result.k, 3.0)
        self.assertEqual(result.seed, 42)
        self.assertTrue(result.deterministic)

    def test_explicit_cli_value_is_preserved(self):
        args = Namespace(
            config="configs/reference_experiment_legacy.yaml",
            epochs=7,
            batch_size=8,
            lr=2e-4,
            pos_weight=1.0,
            k=2.0,
            seed=123,
            deterministic=False,
            no_augment=True,
        )
        result = apply_config_defaults(args)
        self.assertEqual((result.epochs, result.batch_size), (7, 8))
        self.assertEqual((result.lr, result.pos_weight), (2e-4, 1.0))
        self.assertEqual((result.k, result.seed), (2.0, 123))
        self.assertTrue(result.no_augment)


if __name__ == "__main__":
    unittest.main()
