"""Machine-readable experiment metadata and derived-record persistence."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass
class RunMetadata:
    run_id: str
    model: str
    model_variant: str
    dataset: str
    seed: int
    intervention: str = "native"
    model_revision: str | None = None
    dataset_revision: str | None = None
    tokenizer: str | None = None
    context_length: int | None = None
    batch_size: int | None = None
    hook_locations: dict[str, str] = field(default_factory=dict)
    gate_tensor_semantics: str | None = None
    dtype: str | None = None
    device: str | None = None
    commit_hash: str | None = None
    repository_dirty: bool | None = None
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    transformers_version: str | None = None
    platform: str = field(default_factory=platform.platform)
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def collect(cls, **kwargs: Any) -> "RunMetadata":
        """Collect stable runtime metadata without serializing secrets."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            commit = None
            dirty = None
        try:
            import transformers

            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = None
        environment_keys = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")
        kwargs.setdefault("commit_hash", commit)
        kwargs.setdefault("repository_dirty", dirty)
        kwargs.setdefault("transformers_version", transformers_version)
        kwargs.setdefault(
            "environment", {key: os.environ[key] for key in environment_keys if key in os.environ}
        )
        return cls(**kwargs)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def save_manifest(path: str | Path, metadata: RunMetadata, config: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": _plain(metadata), "config": _plain(config), "argv": sys.argv}
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return path


def save_records_csv(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write small paper-facing summaries with a stable union of columns."""
    path = Path(path)
    rows = [{key: _plain(value) for key, value in row.items()} for row in records]
    if not rows:
        raise ValueError("cannot persist an empty record sequence")
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_records_parquet(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write large derived tables; fail clearly when the optional backend is absent."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet output requires the 'analysis' optional dependencies") from exc
    path = Path(path)
    rows = [{key: _plain(value) for key, value in row.items()} for row in records]
    if not rows:
        raise ValueError("cannot persist an empty record sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
