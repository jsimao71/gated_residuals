"""Persist the Cycle B9 conditional Gemma stop without fabricating model results."""

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
    parser.add_argument("--config", default="configs/b9_gemma.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    qwen = json.loads(Path(config["qwen_summary"]).read_text(encoding="utf-8"))
    llama = json.loads(Path(config["llama_summary"]).read_text(encoding="utf-8"))
    qwen_end_to_end = bool(qwen["b8_b9_gate_open"])
    qwen_resources = bool(qwen["resource_audit"]["exact_gated_load_eligible"])
    llama_completed = bool(llama["metrics_available"] and llama["models_evaluated"] > 0)
    eligible = bool(qwen_end_to_end and qwen_resources and llama_completed)
    if eligible:
        raise RuntimeError("B9 gate opened; select and preregister a Gemma checkpoint before running")
    output = Path(args.output or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "config_snapshot.yaml")
    decision = {
        "cycle": "B",
        "experiment": "B9",
        "revision": int(config["revision"]),
        "status": "not_run_upstream_qwen_and_llama_gates_closed",
        "intended_model_family": config["intended_model_family"],
        "qwen_end_to_end": qwen_end_to_end,
        "qwen_resource_eligible": qwen_resources,
        "llama_replication_completed": llama_completed,
        "eligible": eligible,
        "checkpoint_selected": False,
        "models_downloaded": 0,
        "models_evaluated": 0,
        "examples_evaluated": 0,
        "metrics_available": False,
        "decision": "do_not_select_or_download_gemma_checkpoint_until_b7_and_b8_complete",
    }
    (output / "summary.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame([decision])
    table.to_csv(output / "decision_table.csv", index=False)
    table.to_parquet(output / "decision_record.parquet", index=False)
    notes = (
        "# B9 failure and null-result notes\n\n"
        "B9 is conditional on an end-to-end Qwen comparison and a completed Llama replication. "
        "B7 did not complete the exact gated-Qwen path, and B8 consequently evaluated no Llama "
        "model. No Gemma checkpoint was selected, downloaded, or evaluated. This is an upstream "
        "eligibility stop, not a Gemma result or a cross-family comparison.\n"
    )
    (output / "failure_null_notes.md").write_text(notes, encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="cycle-b-b9-eligibility",
        model="not_selected",
        model_variant="gemma_conditional_stop",
        dataset="none",
        seed=0,
        intervention="none",
        device="not_allocated",
    )
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(
        "B9 stopped: upstream B7 Qwen and B8 Llama gates are closed.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
