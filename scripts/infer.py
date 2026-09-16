#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POCT clinical end-to-end inference.

Pipeline:
original image
-> 5x5 splitting
-> RGB preprocessing
-> reference-map generation
-> AttentionResNet34 inference
-> patch-level predictions
-> source-image summaries
-> subfolder summaries

The original project and original images are read-only.
All generated files are written beneath --output_dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from config import load_config


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POCT 5x5 clinical image detection"
    )

    parser.add_argument(
        "--project_root",
        required=True,
        help="Original Siamese project root; read-only",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Original clinical-image batch directory",
    )
    parser.add_argument(
        "--weights_path",
        required=True,
        help="Fixed AttentionResNet34 checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="New result directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Patch-level positive-class (P) classification threshold",
    )
    parser.add_argument(
        "--t_threshold",
        type=float,
        default=0.088,
        help="Positive patch-ratio threshold",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help="Reference-map threshold coefficient",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration to validate and archive.",
    )

    return parser.parse_args()


def write_run_provenance(args, output_dir: Path, weights_path: Path) -> None:
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        load_config(config_path)
        (output_dir / "config_used.yaml").write_bytes(config_path.read_bytes())

    (output_dir / "checkpoint_sha256.txt").write_text(
        f"{sha256_file(weights_path)}  {weights_path.name}\n",
        encoding="utf-8",
    )
    environment = {
        "algorithm_name": "Siamese",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "threshold": args.threshold,
        "positive_patch_ratio_threshold": args.t_threshold,
        "reference_map_K": args.k,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def list_images(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def import_project_modules(project_root: Path):
    try:
        from data.image_splitter import process_folder_structure
        from models.attention_resnet34 import AttentionResNet34
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import Siamese release modules: {exc}"
        ) from exc

    return process_folder_structure, AttentionResNet34


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)!r}"
        )

    if isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint.get("model_state_dict"), dict):
        checkpoint = checkpoint["model_state_dict"]

    state_dict: Dict[str, torch.Tensor] = {}

    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue

        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = value

    if not state_dict:
        raise ValueError("No tensor state_dict entries found")

    return state_dict


def load_model(
    model_class,
    weights_path: Path,
    device: torch.device,
):
    model = model_class(num_class=1).to(device)

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
        )

    state_dict = extract_state_dict(checkpoint)

    # 必须完整、严格加载，禁止静默跳过参数
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    return model, len(state_dict)


def create_reference_tensor(
    image: Image.Image,
    k: float,
) -> torch.Tensor:
    """
    Paper-consistent reference-map preprocessing:

    RGB
    -> grayscale
    -> threshold = global mean - K * global standard deviation
    -> cv2.THRESH_BINARY_INV
    -> nearest-neighbor resize to 224x224
    -> [0,1] tensor
    """

    rgb_array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    gray = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2GRAY,
    )

    threshold_value = float(
        gray.mean() - k * gray.std()
    )

    _, binary = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV,
    )

    reference_image = Image.fromarray(
        binary.astype(np.uint8),
    )

    reference_image = reference_image.resize(
        (224, 224),
        resample=Image.Resampling.NEAREST,
    )

    reference_tensor = transforms.ToTensor()(
        reference_image
    )

    return reference_tensor


def parse_patch_filename(
    patch_path: Path,
) -> Tuple[str, int, int]:
    match = re.match(
        r"^(.*)_block_(\d+)_(\d+)$",
        patch_path.stem,
    )

    if match is None:
        return patch_path.stem, -1, -1

    return (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
    )


class ClinicalPatchDataset(Dataset):
    def __init__(
        self,
        patch_root: Path,
        patch_paths: List[Path],
        k: float,
    ):
        self.patch_root = patch_root
        self.patch_paths = patch_paths
        self.k = k

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, index: int):
        patch_path = self.patch_paths[index]

        with Image.open(patch_path) as handle:
            image = handle.convert("RGB")

            rgb_tensor = self.rgb_transform(image)

            reference_tensor = create_reference_tensor(
                image,
                self.k,
            )

        relative_path = patch_path.relative_to(
            self.patch_root
        )

        relative_folder = str(relative_path.parent)

        source_stem, row, column = parse_patch_filename(
            patch_path
        )

        return {
            "rgb": rgb_tensor,
            "reference": reference_tensor,
            "relative_folder": relative_folder,
            "patch_file": str(relative_path),
            "source_stem": source_stem,
            "row": row,
            "column": column,
        }


def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
    classification_threshold: float,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    processed = 0
    total = len(loader.dataset)

    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(
                device,
                non_blocking=True,
            )

            reference = batch["reference"].to(
                device,
                non_blocking=True,
            )

            logits = model(
                rgb,
                reference,
            ).reshape(-1)

            probabilities = torch.sigmoid(
                logits
            ).detach().cpu().numpy()

            batch_size = len(probabilities)

            for index in range(batch_size):
                probability = float(
                    probabilities[index]
                )

                prediction = int(
                    probability > classification_threshold
                )

                records.append(
                    {
                        "relative_folder":
                            batch["relative_folder"][index],
                        "source_image":
                            batch["source_stem"][index],
                        "patch_file":
                            batch["patch_file"][index],
                        "row":
                            int(batch["row"][index]),
                        "column":
                            int(batch["column"][index]),
                        "probability_P":
                            probability,
                        "prediction":
                            prediction,
                        "predicted_label":
                            "P" if prediction == 1 else "N",
                        "classification_threshold":
                            classification_threshold,
                    }
                )

            processed += batch_size

            print(
                f"\rInference: {processed}/{total} patches",
                end="",
                flush=True,
            )

    print()

    return pd.DataFrame.from_records(records)


def summarize_predictions(
    patch_df: pd.DataFrame,
    group_columns: List[str],
    positive_ratio_threshold: float,
) -> pd.DataFrame:
    summary_df = (
        patch_df
        .groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            total_patches=(
                "prediction",
                "size",
            ),
            p_patches=(
                "prediction",
                "sum",
            ),
        )
        .reset_index()
    )

    summary_df["n_patches"] = (
        summary_df["total_patches"]
        - summary_df["p_patches"]
    )

    summary_df["p_patch_ratio"] = (
        summary_df["p_patches"]
        / summary_df["total_patches"]
    )

    summary_df["n_patch_ratio"] = (
        summary_df["n_patches"]
        / summary_df["total_patches"]
    )

    summary_df[
        "positive_ratio_threshold"
    ] = positive_ratio_threshold

    # 与之前补充实验规则保持一致：严格大于阈值
    summary_df["is_positive"] = (
        summary_df["p_patch_ratio"]
        > positive_ratio_threshold
    )

    summary_df["final_label"] = np.where(
        summary_df["is_positive"],
        "positive",
        "negative",
    )

    # Keep the exported schema stable and place the complementary P/N counts
    # and ratios together for straightforward review and downstream analysis.
    ordered_columns = [
        *group_columns,
        "total_patches",
        "p_patches",
        "n_patches",
        "p_patch_ratio",
        "n_patch_ratio",
        "positive_ratio_threshold",
        "is_positive",
        "final_label",
    ]

    return summary_df[ordered_columns]


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    original_image_count: int,
    patch_count: int,
    state_tensor_count: int,
    weights_sha256: str,
    device: torch.device,
    patch_df: pd.DataFrame,
    image_df: pd.DataFrame,
    folder_df: pd.DataFrame,
) -> None:
    report_path = output_dir / "detection_report.txt"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "POCT end-to-end clinical detection report\n"
        )
        handle.write("=" * 64 + "\n")

        handle.write(
            f"Generated: "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
        )
        handle.write(
            f"Input directory: {args.input_dir}\n"
        )
        handle.write(
            f"Checkpoint: {args.weights_path}\n"
        )
        handle.write(
            f"Checkpoint SHA-256: {weights_sha256}\n"
        )
        handle.write(
            f"Strictly loaded state tensors: "
            f"{state_tensor_count}\n"
        )
        handle.write(
            f"Device: {device}\n"
        )
        handle.write(
            f"K: {args.k}\n"
        )
        handle.write(
            f"Patch threshold: {args.threshold}\n"
        )
        handle.write(
            f"Positive patch-ratio threshold: "
            f"{args.t_threshold}\n"
        )
        handle.write(
            f"Original image count: "
            f"{original_image_count}\n"
        )
        handle.write(
            f"Expected 5x5 patch count: "
            f"{original_image_count * 25}\n"
        )
        handle.write(
            f"Actual patch count: {patch_count}\n"
        )
        handle.write(
            f"Patch prediction rows: "
            f"{len(patch_df)}\n"
        )
        handle.write(
            f"Source-image summaries: "
            f"{len(image_df)}\n"
        )
        handle.write(
            f"Subfolder summaries: "
            f"{len(folder_df)}\n\n"
        )

        handle.write(
            "Model: AttentionResNet34 with attention "
            "inserted after layer3 and layer4.\n"
        )
        handle.write(
            "Checkpoint role: fixed checkpoint retained "
            "from the preceding experiments for supplementary "
            "analysis consistency.\n"
        )
        handle.write(
            "Reference maps: global grayscale mean - "
            "K * standard deviation, THRESH_BINARY_INV, "
            "nearest-neighbor resize.\n"
        )
        handle.write(
            "RGB preprocessing: bilinear resize to 224x224 "
            "and ImageNet normalization.\n"
        )


def main() -> None:
    args = parse_args()

    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    input_dir = Path(
        args.input_dir
    ).expanduser().resolve()

    weights_path = Path(
        args.weights_path
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    if not project_root.is_dir():
        raise FileNotFoundError(
            f"Project root not found: {project_root}"
        )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    # 禁止把输出写回原始项目目录
    try:
        output_dir.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output_dir must not be inside the original project"
        )

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_run_provenance(args, output_dir, weights_path)

    split_dir = output_dir / "split_images"

    split_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        process_folder_structure,
        AttentionResNet34,
    ) = import_project_modules(project_root)

    original_images = list_images(input_dir)

    original_image_count = len(original_images)

    if original_image_count == 0:
        raise RuntimeError(
            f"No supported images found in {input_dir}"
        )

    print(
        f"Original images: {original_image_count}"
    )
    print(
        f"Expected patches: {original_image_count * 25}"
    )
    print(
        f"Split output: {split_dir}"
    )

    process_folder_structure(
        str(input_dir),
        str(split_dir),
    )

    patch_paths = list_images(split_dir)

    patch_count = len(patch_paths)

    expected_patch_count = (
        original_image_count * 25
    )

    print(
        f"Actual patches: {patch_count}"
    )

    if patch_count != expected_patch_count:
        raise RuntimeError(
            "Patch-count audit failed: "
            f"expected {expected_patch_count}, "
            f"found {patch_count}. "
            "Inference was not started."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model, state_tensor_count = load_model(
        AttentionResNet34,
        weights_path,
        device,
    )

    print(
        "Strict checkpoint loading: PASS"
    )
    print(
        f"Loaded state tensors: {state_tensor_count}"
    )

    dataset = ClinicalPatchDataset(
        split_dir,
        patch_paths,
        args.k,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    patch_df = run_inference(
        model,
        loader,
        device,
        args.threshold,
    )

    source_image_df = summarize_predictions(
        patch_df,
        [
            "relative_folder",
            "source_image",
        ],
        args.t_threshold,
    )

    folder_df = summarize_predictions(
        patch_df,
        ["relative_folder"],
        args.t_threshold,
    )

    patch_csv = (
        output_dir
        / "patch_predictions.csv"
    )

    source_image_csv = (
        output_dir
        / "source_image_predictions.csv"
    )

    folder_csv = (
        output_dir
        / "folder_predictions.csv"
    )

    patch_df.to_csv(
        patch_csv,
        index=False,
        encoding="utf-8-sig",
    )

    source_image_df.to_csv(
        source_image_csv,
        index=False,
        encoding="utf-8-sig",
    )

    folder_df.to_csv(
        folder_csv,
        index=False,
        encoding="utf-8-sig",
    )

    try:
        excel_path = (
            output_dir
            / "clinical_detection_summary.xlsx"
        )

        with pd.ExcelWriter(excel_path) as writer:
            folder_df.to_excel(
                writer,
                sheet_name="Folder summary",
                index=False,
            )

            source_image_df.to_excel(
                writer,
                sheet_name="Source image summary",
                index=False,
            )

            patch_df.to_excel(
                writer,
                sheet_name="Patch predictions",
                index=False,
            )

    except Exception as exc:
        print(
            f"Warning: Excel output skipped: {exc}"
        )

    weights_sha256 = sha256_file(
        weights_path
    )

    run_config = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
        "project_root":
            str(project_root),
        "input_dir":
            str(input_dir),
        "output_dir":
            str(output_dir),
        "split_dir":
            str(split_dir),
        "model":
            "models.attention_resnet34.AttentionResNet34",
        "weights_path":
            str(weights_path),
        "weights_sha256":
            weights_sha256,
        "strict_checkpoint_loading":
            True,
        "grid":
            "5x5",
        "expected_patches_per_image":
            25,
        "rgb_preprocessing": {
            "image_size":
                [224, 224],
            "resize_interpolation":
                "bilinear",
            "mean":
                [0.485, 0.456, 0.406],
            "std":
                [0.229, 0.224, 0.225],
        },
        "reference_map": {
            "K":
                args.k,
            "formula":
                "mean - K * standard deviation",
            "threshold":
                "cv2.THRESH_BINARY_INV",
            "resize_interpolation":
                "nearest-neighbor",
            "range":
                [0, 1],
        },
        "patch_classification_threshold":
            args.threshold,
        "positive_patch_ratio_threshold":
            args.t_threshold,
        "positive_rule":
            "p_patch_ratio > threshold",
        "original_image_count":
            original_image_count,
        "actual_patch_count":
            patch_count,
        "device":
            str(device),
        "software": {
            "python":
                platform.python_version(),
            "torch":
                torch.__version__,
            "torchvision":
                torchvision.__version__,
            "opencv":
                cv2.__version__,
            "numpy":
                np.__version__,
            "pandas":
                pd.__version__,
            "pillow":
                Image.__version__,
        },
    }

    with (
        output_dir / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_config,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    write_report(
        output_dir,
        args,
        original_image_count,
        patch_count,
        state_tensor_count,
        weights_sha256,
        device,
        patch_df,
        source_image_df,
        folder_df,
    )

    checksum_lines = []

    for path in sorted(output_dir.iterdir()):
        if (
            path.is_file()
            and path.name != "SHA256SUMS"
        ):
            checksum_lines.append(
                f"{sha256_file(path)}  {path.name}"
            )

    (
        output_dir / "SHA256SUMS"
    ).write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Detection completed successfully.")
    print(f"Patch CSV: {patch_csv}")
    print(
        f"Source-image CSV: {source_image_csv}"
    )
    print(f"Folder CSV: {folder_csv}")
    print(
        f"Report: "
        f"{output_dir / 'detection_report.txt'}"
    )


if __name__ == "__main__":
    main()
