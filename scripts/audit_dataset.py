"""Audit fixed image-patch splits for integrity and potential leakage.

This tool is read-only with respect to the dataset. Reports may contain source
identifiers derived from filenames and must be treated as internal until they
have passed a privacy review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
LABELS = ("T", "UT")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PATCH_PATTERN = re.compile(r"^(?P<source>.+)_block_(?P<row>[0-4])_(?P<column>[0-4])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train/val/test image-patch splits for integrity, exact "
            "duplicates, source-group overlap, and perceptual-hash candidates."
        )
    )
    parser.add_argument(
        "--data_root",
        required=True,
        type=Path,
        help="Dataset root containing train, val, and test directories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="New directory for internal audit reports.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def split_names(records: List[dict]) -> str:
    return ";".join(sorted({record["split"] for record in records}))


def label_names(records: List[dict]) -> str:
    return ";".join(sorted({record["label"] for record in records}))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing audit directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    records: List[dict] = []
    errors: List[dict] = []
    counts: Counter = Counter()
    dimensions: Counter = Counter()
    modes: Counter = Counter()

    for split in SPLITS:
        for label in LABELS:
            class_dir = data_root / split / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Required class directory missing: {class_dir}")

            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            counts[(split, label)] = len(paths)

            for index, path in enumerate(paths, start=1):
                match = PATCH_PATTERN.match(path.stem)
                source_id = match.group("source") if match else ""
                row = int(match.group("row")) if match else ""
                column = int(match.group("column")) if match else ""

                try:
                    with Image.open(path) as opened:
                        opened.load()
                        image = opened.convert("RGB")
                        width, height = image.size
                        mode = opened.mode
                        file_format = opened.format or ""
                        dhash = difference_hash(image)
                        thumbnail = image.convert("L").resize(
                            (32, 32), Image.Resampling.BILINEAR
                        )
                        brightness = float(
                            np.asarray(thumbnail, dtype=np.float32).mean()
                        )

                    digest = sha256_file(path)
                    dimensions[(split, label, width, height)] += 1
                    modes[(split, label, mode, file_format)] += 1
                    records.append(
                        {
                            "split": split,
                            "label": label,
                            "filename": path.name,
                            "source_id": source_id,
                            "patch_row": row,
                            "patch_column": column,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "file_format": file_format,
                            "file_size_bytes": path.stat().st_size,
                            "sha256": digest,
                            "dhash64": dhash,
                            "thumbnail_mean_intensity": f"{brightness:.6f}",
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "split": split,
                            "label": label,
                            "filename": path.name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

                if index % 1000 == 0:
                    print(f"Audited {split}/{label}: {index}/{len(paths)}")

    by_source: Dict[str, List[dict]] = defaultdict(list)
    by_sha256: Dict[str, List[dict]] = defaultdict(list)
    by_dhash: Dict[str, List[dict]] = defaultdict(list)
    brightness_groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for record in records:
        if record["source_id"]:
            by_source[record["source_id"]].append(record)
        by_sha256[record["sha256"]].append(record)
        by_dhash[record["dhash64"]].append(record)
        brightness_groups[(record["split"], record["label"])].append(
            float(record["thumbnail_mean_intensity"])
        )

    source_overlaps = []
    for source_id, group in sorted(by_source.items()):
        represented_splits = {record["split"] for record in group}
        if len(represented_splits) <= 1:
            continue
        source_overlaps.append(
            {
                "source_id": source_id,
                "represented_splits": split_names(group),
                "represented_labels": label_names(group),
                "train_patch_count": sum(r["split"] == "train" for r in group),
                "val_patch_count": sum(r["split"] == "val" for r in group),
                "test_patch_count": sum(r["split"] == "test" for r in group),
                "total_patch_count": len(group),
            }
        )

    exact_overlaps = []
    for digest, group in sorted(by_sha256.items()):
        represented_splits = {record["split"] for record in group}
        if len(group) > 1:
            exact_overlaps.append(
                {
                    "sha256": digest,
                    "represented_splits": split_names(group),
                    "represented_labels": label_names(group),
                    "cross_split": len(represented_splits) > 1,
                    "file_count": len(group),
                    "files": "|".join(
                        f'{r["split"]}/{r["label"]}/{r["filename"]}' for r in group
                    ),
                }
            )

    perceptual_candidates = []
    for digest, group in sorted(by_dhash.items()):
        represented_splits = {record["split"] for record in group}
        exact_hashes = {record["sha256"] for record in group}
        if len(represented_splits) > 1 and len(exact_hashes) > 1:
            perceptual_candidates.append(
                {
                    "dhash64": digest,
                    "represented_splits": split_names(group),
                    "represented_labels": label_names(group),
                    "file_count": len(group),
                    "files": "|".join(
                        f'{r["split"]}/{r["label"]}/{r["filename"]}' for r in group
                    ),
                }
            )

    unmatched = [record for record in records if not record["source_id"]]
    test_source_overlaps = [row for row in source_overlaps if row["test_patch_count"]]

    summary = {
        "schema_version": 1,
        "data_root": str(data_root),
        "reports_contain_internal_filenames": True,
        "total_image_files": sum(counts.values()),
        "successfully_audited_files": len(records),
        "unreadable_files": len(errors),
        "class_counts": {
            split: {label: counts[(split, label)] for label in LABELS}
            for split in SPLITS
        },
        "source_group_count": len(by_source),
        "filenames_not_matching_patch_pattern": len(unmatched),
        "cross_split_source_group_count": len(source_overlaps),
        "test_overlapping_source_group_count": len(test_source_overlaps),
        "test_patches_in_cross_split_source_groups": sum(
            row["test_patch_count"] for row in test_source_overlaps
        ),
        "duplicate_sha256_group_count": len(exact_overlaps),
        "cross_split_exact_duplicate_group_count": sum(
            row["cross_split"] for row in exact_overlaps
        ),
        "cross_split_equal_dhash_candidate_group_count": len(perceptual_candidates),
        "mean_thumbnail_intensity": {
            split: {
                label: round(mean(brightness_groups[(split, label)]), 6)
                for label in LABELS
            }
            for split in SPLITS
        },
    }

    write_csv(
        output_dir / "file_audit.csv",
        records[0].keys() if records else [],
        records,
    )
    write_csv(
        output_dir / "source_group_overlaps.csv",
        source_overlaps[0].keys() if source_overlaps else (
            "source_id",
            "represented_splits",
            "represented_labels",
            "train_patch_count",
            "val_patch_count",
            "test_patch_count",
            "total_patch_count",
        ),
        source_overlaps,
    )
    write_csv(
        output_dir / "exact_duplicate_groups.csv",
        exact_overlaps[0].keys() if exact_overlaps else (
            "sha256",
            "represented_splits",
            "represented_labels",
            "cross_split",
            "file_count",
            "files",
        ),
        exact_overlaps,
    )
    write_csv(
        output_dir / "perceptual_hash_candidates.csv",
        perceptual_candidates[0].keys() if perceptual_candidates else (
            "dhash64",
            "represented_splits",
            "represented_labels",
            "file_count",
            "files",
        ),
        perceptual_candidates,
    )
    write_csv(
        output_dir / "unmatched_filenames.csv",
        records[0].keys() if records else [],
        unmatched,
    )
    write_csv(
        output_dir / "read_errors.csv",
        ("split", "label", "filename", "error_type", "error"),
        errors,
    )
    write_csv(
        output_dir / "dimension_counts.csv",
        ("split", "label", "width", "height", "image_count"),
        (
            {
                "split": key[0],
                "label": key[1],
                "width": key[2],
                "height": key[3],
                "image_count": value,
            }
            for key, value in sorted(dimensions.items())
        ),
    )

    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
