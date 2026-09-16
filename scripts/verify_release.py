"""Verify that the Siamese release package meets its pre-archive checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


EXPECTED_CHECKPOINT = "bc37d1f338c0811d4220bfbceaf7b955c77f89d20b18902e82b2a1baec21599a"
CHECKPOINT = Path("model_artifacts/siamese_attention_resnet34_supplementary_checkpoint.pth")
REQUIRED_FILES = (
    Path("README.md"),
    Path("requirements.txt"),
    Path("configs/release_supplementary_inference.yaml"),
    CHECKPOINT,
    Path("tests/test_release_smoke.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the Siamese release package.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results = []

    for required in REQUIRED_FILES:
        path = root / required
        results.append(check(
            f"required:{required.as_posix()}",
            path.is_file(),
            "present" if path.is_file() else "missing",
        ))

    yaml_files = sorted((root / "configs").glob("*.yaml"))
    yaml_ok = True
    for path in yaml_files:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("top-level YAML value is not a mapping")
            if "project" in config:
                algorithm = config["project"].get("algorithm_name")
                if algorithm != "Siamese":
                    raise ValueError(f"algorithm_name={algorithm!r}")
        except Exception as exc:
            yaml_ok = False
            results.append(check(f"yaml:{path.name}", False, str(exc)))
        else:
            results.append(check(f"yaml:{path.name}", True, "parsed and identity verified"))
    results.append(check("yaml:all", yaml_ok, f"checked {len(yaml_files)} files"))

    checkpoint_path = root / CHECKPOINT
    actual_hash = sha256(checkpoint_path) if checkpoint_path.is_file() else ""
    results.append(check(
        "checkpoint:sha256",
        actual_hash == EXPECTED_CHECKPOINT,
        actual_hash or "checkpoint missing",
    ))

    files = [path for path in root.rglob("*") if path.is_file()]
    dataset_files = [path for path in files if "data/SimoLFA_Dataset" in path.relative_to(root).as_posix()]
    results.append(check("scope:dataset_manifest_present", (root / "data/SimoLFA_Dataset/metadata/dataset_manifest.csv").is_file(), f"{len(dataset_files)} dataset files"))

    bad_names = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if re.search(r"[\s()]|[^\x00-\x7F]", relative):
            bad_names.append(relative)
    results.append(check("names:ascii_no_spaces", not bad_names, "|".join(bad_names[:20]) or "clean"))

    caches = [
        path.relative_to(root).as_posix()
        for path in files
        if path.suffix == ".pyc" or "__pycache__" in path.parts
    ]
    results.append(check("build:no_python_cache", not caches, "|".join(caches[:20]) or "clean"))

    absolute_path_hits = []
    for path in files:
        if path.suffix.lower() not in {".md", ".txt", ".yaml", ".yml", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(?:[A-Za-z]:\\|/home/|/mnt/)", text):
            absolute_path_hits.append(path.relative_to(root).as_posix())
    results.append(check("docs:no_machine_absolute_paths", not absolute_path_hits, "|".join(absolute_path_hits[:20]) or "clean"))

    help_results = []
    for script in ("train.py", "evaluate.py", "infer.py", "audit_dataset.py"):
        command = [sys.executable, "-m", f"scripts.{script[:-3]}", "--help"]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        passed = completed.returncode == 0
        help_results.append(passed)
        results.append(check(f"cli:{script}", passed, "--help passed" if passed else completed.stderr.strip()))
    results.append(check("cli:all", all(help_results), "formal entry points"))

    report = {
        "schema_version": 1,
        "algorithm_name": "Siamese",
        "package_root": str(root),
        "checks": results,
        "passed": all(item["passed"] for item in results),
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
