"""Evaluate the preregistered E6 hard-skipping eligibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e6_hard_skip.yaml")
    parser.add_argument("--output", default="results/e6")
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    config = read_yaml(args.config)
    sources = {name: load(path) for name, path in config["sources"].items()}
    e1, e3, e5 = sources["e1"], sources["e3"], sources["e5"]
    gated_effect = e3["gated_minus_baseline_accuracy"]
    evidence = {
        "strong_interference_supported": bool(e1["strong_interference_supported"]),
        "gated_accuracy_ci_excludes_zero": bool(gated_effect["seed_ci_low"] > 0 or gated_effect["seed_ci_high"] < 0),
        "fraction_gates_below_0_9": float(e3["fraction_gate_below_0_9"]),
        "e1_aligned_suppression_supported": bool(e5["e1_aligned_suppression_supported"]),
    }
    eligibility = config["eligibility"]
    phenomenon = evidence["strong_interference_supported"] or evidence["gated_accuracy_ci_excludes_zero"]
    selective = evidence["fraction_gates_below_0_9"] >= float(eligibility["minimum_fraction_gates_below_0_9"])
    eligible = phenomenon and selective and evidence["e1_aligned_suppression_supported"]
    summary = {
        "stage": "E6", "status": "eligible" if eligible else "not_run_preregistered_stop",
        "eligible": eligible, "evidence": evidence,
        "decision": "run_threshold_and_lambda_sweep" if eligible else config["decision_if_ineligible"],
        "scientific_interpretation": (
            "Hard-skipping sweep authorized by prior evidence."
            if eligible else
            "Soft gates provided no selective-inhibition or quality signal; thresholding them would be an unmotivated post-hoc experiment."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="e6-eligibility", model="tiny_residual_decoder", model_variant="eligibility_audit",
        dataset="synthetic_counterfactual_v1", seed=-1, intervention="none", device="not_applicable",
    )
    save_manifest(output / "manifest.json", metadata, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
