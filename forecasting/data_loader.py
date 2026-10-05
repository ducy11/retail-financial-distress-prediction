"""Load the prepared splits and the manifest from plain JSON, with no pandas dependency.

Each sample carries `request.history` of past quarters plus `as_of` and the target period, which is all
the downstream feature code needs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import MANIFEST_FILE, PREPARED_FILES, SUFFIX, TARGET, manifest_file, prepared_files


def load_manifest(path: Path | None = None) -> Dict[str, Any]:
    """Read the manifest (split policy, sample stats, source sha256).

    By default reads the manifest for `prepared_dir()` (honors env var
    `FORECASTING_PREPARED_DIR`); pass `path` to point at another file.
    """
    target = path or manifest_file()
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prepared(name: str, prepared_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Load one prepared split: "train" | "validation" | "test" | "purged".

    `prepared_dir` lets you read another split (e.g. `data/prepared-rule`); by default uses
    `config.prepared_dir()` and honors env var `FORECASTING_PREPARED_DIR`.
    """
    files = prepared_files(prepared_dir) if prepared_dir else prepared_files()
    if name not in files:
        raise ValueError(f"Unknown split: {name!r}; have {sorted(files)}")
    path = files[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} (in {path.parent}). Run `python -m forecasting.data` "
            f"to rebuild splits, or `python -m scripts.relabel` to build rule-based splits."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def history_array(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Oldest-to-newest quarter history (raw rows, string VND values or None)."""
    return sample["request"]["history"]


def to_float(value: Any) -> float:
    """Convert a VND value (integer string) or None to float (None becomes NaN)."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def field_names(history_row: Dict[str, Any]) -> List[str]:
    """Names of the 16 indicators present in one history row."""
    return [k for k in history_row if k.startswith("revenue") or k.endswith(SUFFIX)]


def label_of(sample: Dict[str, Any]) -> int:
    """Binary distress label."""
    return int(sample[TARGET])


def to_analytic_columns(history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Convert history into {field: [float, ...]} dict for numpy feature math."""
    cols: Dict[str, List[float]] = {}
    fields = field_names(history[0])
    for field in fields:
        cols[field] = [to_float(row.get(field)) for row in history]
    return cols
