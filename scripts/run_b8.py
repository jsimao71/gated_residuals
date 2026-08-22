"""Persist the Cycle B8 conditional Llama stop without fabricating model results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b8_llama.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    upstream = json.loads(Path(config["upstream_summary"]).read_text(encoding="utf-8"))
    standard_parity = upstream["standard_model"]["native_parity_max_logit_error"] == 0
    gated_parity = bool(upstream["gated_release"]["real_native_parity_completed"])
    manageable = bool(upstream["resource_audit"]["exact_gated_load_eligible"])
    eligible = bool(
        standard_parity
        and gated_parity
        and manageable
        and upstream["b8_b9_gate_open"]
    )
    if eligible:
        raise RuntimeError("B8 gate opened; select and preregister a Llama checkpoint before running")
    output = Path(args.output or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "config_snapshot.yaml")
    decision = {
        "cycle": "B",
        "experiment": "B8",
        "revision": int(config["revision"]),
        "status": "not_run_upstream_qwen_gate_closed",
        "intended_model_family": config["intended_model_family"],
        "standard_qwen_real_parity": standard_parity,
        "gated_qwen_real_parity": gated_parity,
        "gated_qwen_resource_eligible": manageable,
        "b7_end_to_end": bool(upstream["b8_b9_gate_open"]),
        "eligible": eligible,
        "checkpoint_selected": False,
        "models_downloaded": 0,
        "models_evaluated": 0,
        "examples_evaluated": 0,
        "metrics_available": False,
        "decision": "do_not_select_or_download_llama_checkpoint_until_b7_works_end_to_end",
    }
    (output / "summary.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame([decision])
    table.to_csv(output / "decision_table.csv", index=False)
    table.to_parquet(output / "decision_record.parquet", index=False)
    notes = (
        "# B8 failure and null-result notes\n\n"
        "B8 is conditional on an end-to-end B7 Qwen baseline/headwise-gated comparison with real "
        "native parity and manageable resource use. B7 validated the standard-Qwen adapter but "
        "did not load or evaluate the exact gated checkpoint. No Llama checkpoint was selected, "
        "downloaded, or evaluated. This is an upstream eligibility stop, not a Llama result.\n"
    )
    (output / "failure_null_notes.md").write_text(notes, encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="cycle-b-b8-eligibility",
        model="not_selected",
        model_variant="llama_conditional_stop",
        dataset="none",
        seed=0,
        intervention="none",
        device="not_allocated",
    )
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(
        "B8 stopped: upstream B7 exact gated parity/resource gate is closed.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
