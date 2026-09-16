import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import POCTDataset
from data.image_splitter import split_image_5x5
from data.preprocess import attention_branch_preprocess
from models.attention_resnet34 import AttentionResNet34


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "model_artifacts"
    / "siamese_attention_resnet34_supplementary_checkpoint.pth"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "bc37d1f338c0811d4220bfbceaf7b955"
    "c77f89d20b18902e82b2a1baec21599a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseSmokeTests(unittest.TestCase):
    def test_split_image_5x5_preserves_all_pixels(self):
        source_array = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "source.png"
            output_path = temporary_path / "patches"
            Image.fromarray(source_array).save(source_path)

            patch_paths = split_image_5x5(
                str(source_path),
                str(output_path),
                source_path.name,
            )

            self.assertEqual(len(patch_paths), 25)
            rows = []
            for row_index in range(5):
                row_patches = []
                for column_index in range(5):
                    patch_path = output_path / (
                        f"source_block_{row_index}_{column_index}.png"
                    )
                    with Image.open(patch_path) as patch:
                        row_patches.append(np.asarray(patch).copy())
                rows.append(np.concatenate(row_patches, axis=1))

            reconstructed = np.concatenate(rows, axis=0)
            np.testing.assert_array_equal(reconstructed, source_array)

    def test_reference_map_is_binary_and_shape_preserving(self):
        rgb = np.full((19, 23, 3), 200, dtype=np.uint8)
        rgb[5:10, 7:12] = 0
        image = Image.fromarray(rgb)

        reference_map = attention_branch_preprocess(image, K=1.0)
        reference_array = np.asarray(reference_map)

        self.assertEqual(reference_map.mode, "L")
        self.assertEqual(reference_map.size, image.size)
        self.assertTrue(set(np.unique(reference_array)).issubset({0, 255}))

    def test_dataset_loading_and_batch_construction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            for class_name, intensity in (("UT", 220), ("T", 20)):
                class_path = dataset_root / class_name
                class_path.mkdir()
                array = np.full((32, 40, 3), intensity, dtype=np.uint8)
                Image.fromarray(array).save(class_path / "sample.png")

            dataset = POCTDataset(dataset_root, mode="test", K=3.0)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            (rgb_batch, reference_batch), labels = next(iter(loader))

            self.assertEqual(tuple(rgb_batch.shape), (2, 3, 224, 224))
            self.assertEqual(tuple(reference_batch.shape), (2, 1, 224, 224))
            self.assertEqual(labels.tolist(), [0.0, 1.0])
            self.assertTrue(torch.isfinite(rgb_batch).all())
            self.assertTrue(torch.isfinite(reference_batch).all())

    def test_batch_size_one_model_inference(self):
        model = AttentionResNet34(num_class=1).eval()
        rgb = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        reference = torch.zeros((1, 1, 64, 64), dtype=torch.float32)

        with torch.inference_mode():
            output = model(rgb, reference)

        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_checkpoint_checksum_and_strict_loading(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest(f"Checkpoint is not present: {CHECKPOINT_PATH}")

        self.assertEqual(
            sha256_file(CHECKPOINT_PATH),
            EXPECTED_CHECKPOINT_SHA256,
        )

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        if all(torch.is_tensor(value) for value in checkpoint.values()):
            state_dict = checkpoint
        elif isinstance(checkpoint.get("state_dict"), dict):
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint.get("model_state_dict"), dict):
            state_dict = checkpoint["model_state_dict"]
        else:
            self.fail("Checkpoint does not contain a recognized state dictionary")

        model = AttentionResNet34(num_class=1)
        model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    unittest.main()
