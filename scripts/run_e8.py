"""Audit E8's PRA-memory and active-computation axes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e8_pra_factorial.yaml")
    parser.add_argument("--output", default="results/e8")
    return parser.parse_args()


def git_value(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


def main():
    args = parse_args()
    config = read_yaml(args.config)
    root = Path(config["pra_root"])
    requirements = config["requirements"]
    e6 = json.loads(Path(config["sources"]["e6"]).read_text(encoding="utf-8"))
    e7 = json.loads(Path(config["sources"]["e7"]).read_text(encoding="utf-8"))
    code_available = all((root / requirements[name]).exists() for name in ("pra_model_code", "pra_config_code"))
    checkpoints = sorted(root.glob(requirements["routing_checkpoint_glob"]))
    try:
        pra_revision = git_value(root, "rev-parse", "HEAD")
        pra_branch = git_value(root, "branch", "--show-current")
        pra_dirty = bool(git_value(root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        pra_revision, pra_branch, pra_dirty = None, None, None
    memory_axis_available = code_available and bool(checkpoints)
    compute_axis_available = bool(e6["eligible"])
    clean_pinned = pra_revision is not None and pra_dirty is False
    eligible = (
        memory_axis_available and compute_axis_available and
        (clean_pinned or not requirements["require_clean_pinned_pra_checkout"]) and
        (compute_axis_available or not requirements["require_validated_compute_axis"])
    )
    summary = {
        "stage": "E8", "status": "eligible" if eligible else "not_run_preregistered_stop",
        "eligible": eligible,
        "memory_axis": {
            "available": memory_axis_available, "pra_code_available": code_available,
            "routing_checkpoint_count": len(checkpoints), "pra_revision": pra_revision,
            "pra_branch": pra_branch, "pra_worktree_dirty": pra_dirty,
        },
        "compute_axis": {
            "available": compute_axis_available, "e6_status": e6["status"],
            "pretrained_transfer_available": bool(e7["pretrained_metrics_are_available"]),
        },
        "factorial_cells_evaluated": 0,
        "decision": "run_memory_compute_factorial" if eligible else config["decision_if_ineligible"],
        "scientific_interpretation": (
            "Both independently validated axes are available."
            if eligible else
            "PRA memory control exists, but active computation was not validated; crossing it with post-hoc thresholds would not test the registered dual-selection hypothesis."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="e8-eligibility", model="PRA_x_gated_residuals", model_variant="factorial_audit",
        dataset="none", seed=-1, intervention="none", device="not_applicable",
    )
    save_manifest(output / "manifest.json", metadata, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
