"""Configuration loading and validation for the Siamese release package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_PROJECT_FIELDS = {"algorithm_name", "configuration_name", "role"}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration and validate release-level identity fields."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    project = config.get("project")
    if project is not None:
        if not isinstance(project, dict):
            raise ValueError("The project section must be a mapping")
        missing = REQUIRED_PROJECT_FIELDS.difference(project)
        if missing:
            raise ValueError(f"Missing project fields: {sorted(missing)}")
        if project["algorithm_name"] != "Siamese":
            raise ValueError(
                f"algorithm_name must be Siamese, got {project['algorithm_name']!r}"
            )
    return config
