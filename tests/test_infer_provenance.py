import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.infer import summarize_predictions, write_run_provenance


class InferProvenanceTests(unittest.TestCase):
    def test_summary_uses_p_and_n_output_schema(self):
        patch_df = pd.DataFrame(
            {
                "relative_folder": ["sample", "sample", "sample"],
                "prediction": [1, 0, 1],
                "probability_P": [0.9, 0.2, 0.8],
            }
        )

        summary = summarize_predictions(
            patch_df,
            ["relative_folder"],
            positive_ratio_threshold=0.088,
        )

        self.assertEqual(
            list(summary.columns),
            [
                "relative_folder",
                "total_patches",
                "p_patches",
                "n_patches",
                "p_patch_ratio",
                "n_patch_ratio",
                "positive_ratio_threshold",
                "is_positive",
                "final_label",
            ],
        )
        self.assertEqual(int(summary.loc[0, "p_patches"]), 2)
        self.assertEqual(int(summary.loc[0, "n_patches"]), 1)
        self.assertAlmostEqual(summary.loc[0, "p_patch_ratio"], 2 / 3)
        self.assertAlmostEqual(summary.loc[0, "n_patch_ratio"], 1 / 3)

    def test_provenance_files_are_written(self):
        args = Namespace(
            config="configs/release_supplementary_inference.yaml",
            threshold=0.5,
            t_threshold=0.088,
            k=3.0,
            batch_size=32,
            workers=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            write_run_provenance(args, output, checkpoint)
            self.assertTrue((output / "config_used.yaml").is_file())
            self.assertTrue((output / "environment.json").is_file())
            self.assertTrue((output / "checkpoint_sha256.txt").is_file())
            self.assertTrue((output / "command.txt").is_file())


if __name__ == "__main__":
    unittest.main()
