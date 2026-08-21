"""Run E5 combined goal-conditioned gating falsification experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, load_seed, make_loader, resolve_device, train_seed, write_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e5_combined.yaml")
    parser.add_argument("--output", default="results/e5")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


def e1_roles(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {(row["task"], int(row["layer"])): row["classification"] for row in csv.DictReader(stream)}


@torch.inference_mode()
def evaluate_modes(model, loader, device):
    modes = {
        "native": {}, "forced_open": {"gate_mode": "open"},
        "shuffled_goal": {"goal_mode": "shuffled"}, "zero_goal": {"goal_mode": "zero"},
    }
    output = {}
    for name, kwargs in modes.items():
        total, correct, loss_sum = 0, 0, 0.0
        for batch in loader:
            ids, mask, target = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
            logits = model(ids, mask, **kwargs).logits
            loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            correct += int((logits.argmax(-1) == target).sum())
            total += target.numel()
        output[name] = {"accuracy": correct / total, "loss": loss_sum / total, "perplexity": math.exp(min(loss_sum / total, 20))}
    return output


@torch.inference_mode()
def capture_gates(model, loader, examples, device, seed, roles):
    records = []
    model.eval()
    for batch in loader:
        ids, mask, target = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
        native = model(ids, mask, capture=True)
        full_probability = native.logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
        skipped = [model(ids, mask, skip_layers={layer}).logits.softmax(-1).gather(1, target[:, None]).squeeze(1) for layer in range(model.num_layers)]
        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = examples[example_index]
            token_index = int(mask[row].sum()) - 1
            for layer in range(model.num_layers):
                gate = float(native.gates[layer][row])
                candidate_norm = float(torch.linalg.vector_norm(native.candidates[layer][row, token_index]))
                residual_norm = float(torch.linalg.vector_norm(native.states[layer][row, token_index]))
                records.append({
                    "seed": seed, "example_id": example.example_id, "task": example.intent,
                    "layer": layer, "e1_classification": roles[(example.intent, layer)],
                    "gate": gate,
                    "gate_entropy": -(gate * math.log(max(gate, 1e-8)) + (1 - gate) * math.log(max(1 - gate, 1e-8))),
                    "causal_utility": float(full_probability[row] - skipped[layer][row]),
                    "relative_update_norm": candidate_norm / max(residual_norm, 1e-8),
                })
    return records


def prior_quality():
    with Path("results/e3/condition_summary.csv").open(newline="", encoding="utf-8") as stream:
        e3 = list(csv.DictReader(stream))
    with Path("results/e4/summary.json").open(encoding="utf-8") as stream:
        e4 = json.load(stream)
    return {
        "matched_baseline": float(next(row["mean_accuracy"] for row in e3 if row["variant"] == "matched_baseline")),
        "gate_only": float(next(row["mean_accuracy"] for row in e3 if row["variant"] == "gated" and row["condition"] == "native")),
        "goal": float(e4["mean_goal_model_accuracy"]), "goal_only": float(e4["mean_goal_only_accuracy"]),
    }


def main():
    args = parse_args()
    config = read_yaml(args.config)
    base = read_yaml(config["base_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    roles = e1_roles(config["source_e1_summary"])
    quality_rows, gate_rows = [], []
    parameter_count = None
    for seed in base["training"]["seeds"]:
        run_dir = output / "runs" / f"seed-{seed}"
        if args.reuse_checkpoints and (run_dir / "model.pt").exists():
            model, splits, vocabulary, datasets = load_seed(base, run_dir / "model.pt")
        else:
            model, splits, vocabulary, datasets, _ = train_seed(base, "goal_gated", int(seed), run_dir)
        device = resolve_device(base["training"]["device"])
        model.to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        loader = make_loader(datasets["test"], base, shuffle=False, seed=int(seed))
        quality = evaluate_modes(model, loader, device)
        quality_rows.extend({"seed": seed, "condition": condition, **values} for condition, values in quality.items())
        records = capture_gates(model, loader, splits["test"], device, int(seed), roles)
        gate_rows.extend(records)
        write_summary(run_dir / "gate_records.csv", records)
        metadata = RunMetadata.collect(
            run_id=f"e5-goal-gated-seed-{seed}", model="tiny_residual_decoder", model_variant="goal_gated",
            dataset="synthetic_counterfactual_v1", seed=int(seed), context_length=int(base["data"]["max_length"]),
            batch_size=int(base["training"]["batch_size"]), dtype="float32", device=device,
        )
        save_manifest(run_dir / "e5_manifest.json", metadata, {"e5": config, "base": base})
    write_summary(output / "quality_by_seed.csv", quality_rows)
    write_summary(output / "gate_records.csv", gate_rows)
    quality_summary = []
    for condition in config["controls"]:
        values = [row["accuracy"] for row in quality_rows if row["condition"] == condition]
        mean, low, high = confidence(values, seed=505 + len(quality_summary))
        quality_summary.append({"condition": condition, "mean_accuracy": mean, "seed_ci_low": low, "seed_ci_high": high})
    write_summary(output / "quality_summary.csv", quality_summary)
    role_gate = {role: float(np.mean([row["gate"] for row in gate_rows if row["e1_classification"] == role])) for role in sorted({row["e1_classification"] for row in gate_rows})}
    seed_suppression = []
    for seed in base["training"]["seeds"]:
        harmful = [row["gate"] for row in gate_rows if row["seed"] == seed and row["e1_classification"] == "candidate_harmful"]
        useful = [row["gate"] for row in gate_rows if row["seed"] == seed and row["e1_classification"] == "useful"]
        seed_suppression.append(float(np.mean(useful) - np.mean(harmful)))
    suppression_mean, suppression_low, suppression_high = confidence(seed_suppression, seed=505)
    gates = np.asarray([row["gate"] for row in gate_rows])
    utility = np.asarray([row["causal_utility"] for row in gate_rows])
    magnitude = np.asarray([row["relative_update_norm"] for row in gate_rows])
    native_accuracy = next(row["mean_accuracy"] for row in quality_summary if row["condition"] == "native")
    overall = {
        "stage": "E5", "seeds": len(base["training"]["seeds"]), "parameter_count": parameter_count,
        "comparison_accuracy": {**prior_quality(), "goal_gated": native_accuracy},
        "mean_gate": float(gates.mean()), "median_gate": float(np.median(gates)),
        "mean_gate_entropy": float(np.mean([row["gate_entropy"] for row in gate_rows])),
        "mean_gate_by_e1_classification": role_gate,
        "useful_minus_harmful_gate": {"mean": suppression_mean, "seed_ci_low": suppression_low, "seed_ci_high": suppression_high},
        "gate_utility_spearman": float(spearmanr(gates, utility).statistic),
        "gate_relative_magnitude_spearman": float(spearmanr(gates, magnitude).statistic),
        "e1_aligned_suppression_supported": suppression_low > float(config["statistics"]["minimum_suppression_difference"]),
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
