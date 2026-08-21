"""Persist the E7 scientific/resource eligibility audit without fabricating results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e7_transfer.yaml")
    parser.add_argument("--output", default="results/e7")
    parser.add_argument("--cache-root", default=os.environ.get("HF_HOME", "D:/hf-cache-gated-residuals"))
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    config = read_yaml(args.config)
    registered = read_yaml(config["registered_config"])
    evidence = {name: load_json(path) for name, path in config["sources"].items()}
    revision = registered["models"]["gated"]["revision"]
    checkpoint = (
        Path(args.cache_root) / "hub" / "models--QwQZh--gated_attention" /
        "snapshots" / revision / "1B_gate_headwise" / "pytorch_model.bin"
    )
    checkpoint_audit = {"downloaded": checkpoint.exists(), "file_bytes": checkpoint.stat().st_size if checkpoint.exists() else None}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        checkpoint_audit.update({
            "tensor_count": len(state), "named_parameter_count": sum(value.numel() for value in state.values()),
            "dtypes": sorted({str(value.dtype) for value in state.values()}),
            "tied_output_weight_present": "lm_head.weight" in state,
        })
        del state
    device = {
        "cuda_available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "bfloat16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }
    scientific = {
        "strong_interference_supported": bool(evidence["e1"]["strong_interference_supported"]),
        "gated_quality_effect_excludes_zero": bool(evidence["e3"]["gated_minus_baseline_accuracy"]["seed_ci_low"] > 0),
        "shared_goal_control_supported": bool(evidence["e4"]["shared_goal_control_supported"]),
        "e1_aligned_suppression_supported": bool(evidence["e5"]["e1_aligned_suppression_supported"]),
    }
    scientifically_eligible = any(scientific.values())
    resource_eligible = bool(config["load_attempt"]["real_native_parity_completed"])
    summary = {
        "stage": "E7", "status": "not_run_scientific_and_resource_stop",
        "scientifically_eligible": scientifically_eligible, "resource_eligible": resource_eligible,
        "scientific_evidence": scientific, "device": device, "gated_checkpoint": checkpoint_audit,
        "load_attempt": config["load_attempt"], "baseline_checkpoint_downloaded": False,
        "pretrained_examples_evaluated": 0, "pretrained_metrics_are_available": False,
        "decision": config["decision_if_ineligible"],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="e7-eligibility", model=registered["models"]["gated"]["repository"],
        model_variant="1B_gate_headwise_resource_audit", dataset="none", seed=int(registered["seed"]),
        intervention="none", model_revision=revision, dtype="float16_attempted", device=device["name"],
        gate_tensor_semantics=registered["implementation"]["gate_shape"],
    )
    save_manifest(output / "manifest.json", metadata, {"e7": config, "registered": registered})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
