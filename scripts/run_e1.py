"""Run E1 baseline training, exhaustive block ablation, and repair analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.attention_dilution import attention_metrics
from gated_residuals.experiments import confidence, evaluate, load_seed, make_loader, resolve_device, train_seed, write_summary
from gated_residuals.residual_dynamics import update_geometry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--output", default="results/e1")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def capture_records(model, loader, examples, device, seed):
    records = []
    model.eval()
    for batch in loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        target = batch["target"].to(device)
        native = model(ids, mask, capture=True)
        probabilities = native.logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
        skipped = []
        for layer in range(model.num_layers):
            output = model(ids, mask, skip_layers={layer}, capture=True)
            skipped.append((output, output.logits.softmax(-1).gather(1, target[:, None]).squeeze(1)))
        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = examples[example_index]
            for layer in range(model.num_layers):
                token_index = int(mask[row].sum()) - 1
                geometry = update_geometry(
                    native.states[layer][row, token_index], native.candidates[layer][row, token_index]
                )
                attention = attention_metrics(native.attention[layer][row, :, token_index])
                divergences = [0.0]
                skip_output = skipped[layer][0]
                for later in range(layer + 1, model.num_layers + 1):
                    divergences.append(float(torch.linalg.vector_norm(native.states[later][row, token_index] - skip_output.states[later][row, token_index])))
                peak = max(divergences)
                repair_score = (peak - divergences[-1]) / max(peak, 1e-8)
                records.append({
                    "seed": seed, "example_id": example.example_id, "family_id": example.family_id,
                    "task": example.intent, "identifiability": example.identifiability,
                    "goal_identifiability": example.goal_identifiability, "distractor": example.distractor,
                    "layer": layer, "full_probability": float(probabilities[row]),
                    "full_correct": bool(native.logits[row].argmax() == target[row]),
                    "skipped_probability": float(skipped[layer][1][row]),
                    "utility": float(probabilities[row] - skipped[layer][1][row]),
                    "candidate_update_norm": float(geometry["update_norm"]),
                    "relative_update_norm": float(geometry["relative_update_norm"]),
                    "candidate_cosine": float(geometry["state_update_cosine"]),
                    "attention_entropy": float(attention["attention_entropy"].mean()),
                    "attention_effective_support": float(attention["attention_effective_support"].mean()),
                    "repair_score": repair_score,
                    "repair_detected": repair_score >= 0.25,
                })
    return records


def main():
    from gated_residuals.common.config import read_yaml

    args = parse_args()
    config = read_yaml(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_records, seed_rows = [], []
    for seed in config["training"]["seeds"]:
        run_dir = output / "runs" / f"seed-{seed}"
        if args.reuse_checkpoints and (run_dir / "model.pt").exists():
            model, splits, vocabulary, datasets = load_seed(config, run_dir / "model.pt")
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            timing = {"training_seconds": 0.0, "best_val_loss": min(row["val_loss"] for row in history)}
            metadata = RunMetadata.collect(
                run_id=f"baseline-seed-{seed}", model="tiny_residual_decoder",
                model_variant="baseline", dataset="synthetic_counterfactual_v1", seed=int(seed),
                context_length=int(config["data"]["max_length"]),
                batch_size=int(config["training"]["batch_size"]), dtype="float32",
                device=resolve_device(config["training"]["device"]),
            )
            save_manifest(run_dir / "manifest.json", metadata, config)
        else:
            model, splits, vocabulary, datasets, timing = train_seed(config, "baseline", int(seed), run_dir)
        device = resolve_device(config["training"]["device"])
        model.to(device)
        test_loader = make_loader(datasets["test"], config, shuffle=False, seed=int(seed))
        metrics = evaluate(model, test_loader, device)
        records = capture_records(model, test_loader, splits["test"], device, int(seed))
        all_records.extend(records)
        seed_rows.append({"seed": seed, "test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"], **timing})
        write_summary(run_dir / "block_records.csv", records)
    write_summary(output / "block_records.csv", all_records)
    write_summary(output / "seed_summary.csv", seed_rows)

    summaries, seed_summaries = [], []
    threshold = float(config["statistics"]["negligible_utility"])
    for task in sorted({record["task"] for record in all_records}):
        for layer in range(int(config["model"]["layers"])):
            selected = [record for record in all_records if record["task"] == task and record["layer"] == layer]
            seed_means = []
            for seed in config["training"]["seeds"]:
                values = [record["utility"] for record in selected if record["seed"] == seed]
                seed_means.append(sum(values) / len(values))
                example_mean, example_low, example_high = confidence(values, seed=int(seed) + layer)
                seed_summaries.append({
                    "task": task, "layer": layer, "seed": seed,
                    "mean_utility": example_mean, "example_ci_low": example_low,
                    "example_ci_high": example_high,
                })
            mean, low, high = confidence(seed_means, seed=layer)
            negative_seeds = sum(value < -threshold for value in seed_means)
            repair_rate = sum(record["repair_detected"] for record in selected) / len(selected)
            if high < -threshold and negative_seeds >= 4:
                role = "candidate_harmful"
            elif low > threshold:
                role = "useful"
            else:
                role = "redundant_or_uncertain"
            summaries.append({
                "task": task, "layer": layer, "mean_utility": mean, "seed_ci_low": low,
                "seed_ci_high": high, "negative_seed_count": negative_seeds,
                "repair_rate": repair_rate, "classification": role,
            })
    write_summary(output / "layer_task_summary.csv", summaries)
    write_summary(output / "seed_layer_task_summary.csv", seed_summaries)
    overall = {
        "stage": "E1", "seeds": len(seed_rows), "test_examples_per_seed": len(splits["test"]),
        "mean_accuracy": sum(row["test_accuracy"] for row in seed_rows) / len(seed_rows),
        "candidate_harmful_cells": sum(row["classification"] == "candidate_harmful" for row in summaries),
        "useful_cells": sum(row["classification"] == "useful" for row in summaries),
        "uncertain_cells": sum(row["classification"] == "redundant_or_uncertain" for row in summaries),
        "strong_interference_supported": any(row["classification"] == "candidate_harmful" and row["repair_rate"] >= 0.5 for row in summaries),
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
