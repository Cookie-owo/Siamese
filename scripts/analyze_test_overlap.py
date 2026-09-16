"""Stratify saved test predictions by source-group overlap status."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test performance for overlapping and source-independent patches."
    )
    parser.add_argument("--file_audit", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities > threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "sample_count": int(labels.size),
        "negative_count": int((labels == 0).sum()),
        "positive_count": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else None
    )
    result["average_precision"] = (
        float(average_precision_score(labels, probabilities))
        if (labels == 1).any()
        else None
    )
    return result


def main() -> None:
    args = parse_args()
    with args.file_audit.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))

    split_by_source = defaultdict(set)
    source_by_sha256 = defaultdict(set)
    for row in audit_rows:
        if row["source_id"]:
            split_by_source[row["source_id"]].add(row["split"])
            source_by_sha256[row["sha256"]].add(row["source_id"])

    test_rows = [row for row in audit_rows if row["split"] == "test"]
    test_rows.sort(key=lambda row: (0 if row["label"] == "UT" else 1, row["filename"]))

    saved = np.load(args.predictions)
    probabilities = np.asarray(saved["probs"], dtype=np.float64)
    labels = np.asarray(saved["labels"], dtype=np.int64)
    expected_labels = np.asarray(
        [0 if row["label"] == "UT" else 1 for row in test_rows], dtype=np.int64
    )

    if labels.shape != expected_labels.shape or not np.array_equal(labels, expected_labels):
        raise ValueError("Saved prediction order does not match the audited test-file order")

    overlap_mask = []
    overlap_reasons = []
    for row in test_rows:
        related_sources = set()
        if row["source_id"]:
            related_sources.add(row["source_id"])
        related_sources.update(source_by_sha256[row["sha256"]])
        related_splits = set().union(
            *(split_by_source[source] for source in related_sources)
        ) if related_sources else {"test"}
        overlap = bool(related_splits.intersection({"train", "val"}))
        overlap_mask.append(overlap)
        overlap_reasons.append(
            ";".join(sorted(related_splits.intersection({"train", "val"})))
        )

    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    independent_mask = ~overlap_mask

    result = {
        "schema_version": 1,
        "threshold": args.threshold,
        "definition": (
            "A test patch is overlapping when its filename-derived source_id, or "
            "the source_id of an exact duplicate, is present in train or val."
        ),
        "all_test_patches": metrics(labels, probabilities, args.threshold),
        "overlapping_test_patches": metrics(
            labels[overlap_mask], probabilities[overlap_mask], args.threshold
        ),
        "source_independent_test_patches": metrics(
            labels[independent_mask], probabilities[independent_mask], args.threshold
        ),
    }

    independent_sources = defaultdict(list)
    for index, row in enumerate(test_rows):
        if independent_mask[index] and row["source_id"]:
            independent_sources[row["source_id"]].append(index)

    source_labels = []
    source_probabilities = []
    for indices in independent_sources.values():
        group_labels = labels[indices]
        if np.unique(group_labels).size != 1:
            raise ValueError("A source-independent source group has conflicting labels")
        source_labels.append(int(group_labels[0]))
        source_probabilities.append(float(probabilities[indices].mean()))

    result["source_independent_source_groups_mean_probability"] = metrics(
        np.asarray(source_labels, dtype=np.int64),
        np.asarray(source_probabilities, dtype=np.float64),
        args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    detail_path = args.output.with_name("test_overlap_classification.csv")
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("filename", "label", "source_overlap", "overlap_with", "probability")
        )
        for row, overlap, reason, probability in zip(
            test_rows, overlap_mask, overlap_reasons, probabilities
        ):
            writer.writerow(
                (row["filename"], row["label"], bool(overlap), reason, probability)
            )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
