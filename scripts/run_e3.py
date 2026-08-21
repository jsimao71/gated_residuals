"""Run E3 matched scalar residual-gating experiment and controls."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, make_loader, resolve_device, train_seed, write_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e3_gated.yaml")
    parser.add_argument("--output", default="results/e3")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


def ranked_like(values: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Assign the observed marginal values in ascending score order."""
    order = torch.argsort(score.flatten())
    sorted_values = torch.sort(values.flatten()).values
    result = torch.empty_like(sorted_values)
    result[order] = sorted_values
    return result.reshape_as(values)


@torch.inference_mode()
def validation_gate_means(model, loader, examples, device):
    totals = defaultdict(list)
    model.eval()
    for batch in loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        output = model(ids, mask, capture=True)
        for row, example_index in enumerate(batch["example_index"].tolist()):
            task = examples[example_index].intent
            for layer, gate in enumerate(output.gates):
                totals[(task, layer)].append(float(gate[row]))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def make_overrides(condition, output, batch, examples, task_means, generator):
    if condition == "forced_open":
        return [torch.ones_like(gate) for gate in output.gates]
    if condition == "shuffled":
        return [gate.roll(1, dims=0) for gate in output.gates]
    if condition == "random_skip":
        return [torch.bernoulli(gate, generator=generator) for gate in output.gates]
    if condition == "static_task":
        tasks = [examples[index].intent for index in batch["example_index"].tolist()]
        return [torch.tensor([[task_means[(task, layer)]] for task in tasks], device=gate.device) for layer, gate in enumerate(output.gates)]
    if condition in {"candidate_norm", "residual_norm"}:
        overrides = []
        for layer, gate in enumerate(output.gates):
            token_indices = batch["attention_mask"].sum(dim=1).sub(1).to(gate.device)
            source = output.candidates[layer] if condition == "candidate_norm" else output.states[layer]
            rows = torch.arange(source.size(0), device=source.device)
            score = torch.linalg.vector_norm(source[rows, token_indices], dim=-1)
            overrides.append(ranked_like(gate, score))
        return overrides
    raise ValueError(condition)


@torch.inference_mode()
def evaluate_condition(model, loader, examples, device, condition, task_means, seed):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    per_group = defaultdict(lambda: [0, 0])
    generator = torch.Generator(device=device).manual_seed(seed + 9107)
    for batch in loader:
        ids, mask, target = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
        if model.variant == "matched_baseline":
            result = model(ids, mask)
        elif condition == "native":
            result = model(ids, mask)
        else:
            native = model(ids, mask, capture=True)
            overrides = make_overrides(condition, native, batch, examples, task_means, generator)
            result = model(ids, mask, gate_overrides=overrides)
        prediction = result.logits.argmax(-1)
        loss_sum += float(F.cross_entropy(result.logits, target, reduction="sum"))
        correct += int((prediction == target).sum())
        total += target.numel()
        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = examples[example_index]
            for key in (f"task:{example.intent}", f"distractor:{example.distractor}"):
                per_group[key][0] += int(prediction[row] == target[row])
                per_group[key][1] += 1
    return {
        "loss": loss_sum / total, "perplexity": math.exp(min(loss_sum / total, 20)),
        "accuracy": correct / total,
        **{key.replace(":", "_") + "_accuracy": value[0] / value[1] for key, value in per_group.items()},
    }


@torch.inference_mode()
def gate_records(model, loader, examples, device, seed):
    records = []
    model.eval()
    for batch in loader:
        ids, mask, target = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
        native = model(ids, mask, capture=True)
        probability = native.logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
        skipped = [model(ids, mask, skip_layers={layer}).logits.softmax(-1).gather(1, target[:, None]).squeeze(1) for layer in range(model.num_layers)]
        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = examples[example_index]
            token_index = int(mask[row].sum()) - 1
            for layer in range(model.num_layers):
                gate = float(native.gates[layer][row])
                entropy = -(gate * math.log(max(gate, 1e-8)) + (1 - gate) * math.log(max(1 - gate, 1e-8)))
                candidate_norm = float(torch.linalg.vector_norm(native.candidates[layer][row, token_index]))
                residual_norm = float(torch.linalg.vector_norm(native.states[layer][row, token_index]))
                records.append({
                    "seed": seed, "example_id": example.example_id, "family_id": example.family_id,
                    "task": example.intent, "layer": layer, "distractor": example.distractor,
                    "goal_identifiability": example.goal_identifiability, "gate": gate,
                    "gate_entropy": entropy, "candidate_update_norm": candidate_norm,
                    "residual_norm": residual_norm, "relative_update_norm": candidate_norm / max(residual_norm, 1e-8),
                    "causal_utility": float(probability[row] - skipped[layer][row]),
                    "e1_candidate_harmful_cell": example.intent == "sum_mod_10" and layer == 3,
                })
    return records


@torch.inference_mode()
def time_forward(model, loader, device, gate_mode="native", measured=20):
    batches = list(loader)
    for batch in batches[:3]:
        model(batch["input_ids"].to(device), batch["attention_mask"].to(device), gate_mode=gate_mode)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start, examples = time.perf_counter(), 0
    for index in range(measured):
        batch = batches[index % len(batches)]
        model(batch["input_ids"].to(device), batch["attention_mask"].to(device), gate_mode=gate_mode)
        examples += batch["input_ids"].size(0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / examples


def main():
    args = parse_args()
    config = read_yaml(args.config)
    base = read_yaml(config["base_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    condition_rows, all_gate_records = [], []
    parameter_counts = {}
    for variant in config["variants"]:
        for seed in base["training"]["seeds"]:
            run_dir = output / "runs" / variant / f"seed-{seed}"
            checkpoint = run_dir / "model.pt"
            if args.reuse_checkpoints and checkpoint.exists():
                from gated_residuals.experiments import load_seed
                model, splits, vocabulary, datasets = load_seed(base, checkpoint)
                timing = {"training_seconds": 0.0, "best_val_loss": float("nan")}
            else:
                model, splits, vocabulary, datasets, timing = train_seed(base, variant, int(seed), run_dir)
            device = resolve_device(base["training"]["device"])
            model.to(device)
            parameter_counts[variant] = sum(parameter.numel() for parameter in model.parameters())
            val_loader = make_loader(datasets["val"], base, shuffle=False, seed=int(seed))
            test_loader = make_loader(datasets["test"], base, shuffle=False, seed=int(seed))
            task_means = validation_gate_means(model, val_loader, splits["val"], device) if variant == "gated" else {}
            conditions = ["native"] if variant == "matched_baseline" else config["controls"]
            for condition in conditions:
                metrics = evaluate_condition(model, test_loader, splits["test"], device, condition, task_means, int(seed))
                condition_rows.append({"variant": variant, "seed": seed, "condition": condition, **metrics})
            latency_native = time_forward(model, test_loader, device, measured=int(config["timing"]["measured_batches"]))
            condition_rows[-len(conditions)]["milliseconds_per_example"] = latency_native
            if variant == "gated":
                latency_open = time_forward(model, test_loader, device, gate_mode="open", measured=int(config["timing"]["measured_batches"]))
                for row in condition_rows[-len(conditions):]:
                    if row["condition"] == "forced_open":
                        row["milliseconds_per_example"] = latency_open
                records = gate_records(model, test_loader, splits["test"], device, int(seed))
                all_gate_records.extend(records)
                write_summary(run_dir / "gate_records.csv", records)
            metadata = RunMetadata.collect(
                run_id=f"e3-{variant}-seed-{seed}", model="tiny_residual_decoder", model_variant=variant,
                dataset="synthetic_counterfactual_v1", seed=int(seed), context_length=int(base["data"]["max_length"]),
                batch_size=int(base["training"]["batch_size"]), dtype="float32", device=device,
            )
            save_manifest(run_dir / "e3_manifest.json", metadata, {"e3": config, "base": base})
    write_summary(output / "condition_summary_by_seed.csv", condition_rows)
    write_summary(output / "gate_records.csv", all_gate_records)
    aggregate = []
    for variant, condition in sorted({(row["variant"], row["condition"]) for row in condition_rows}):
        selected = [row for row in condition_rows if row["variant"] == variant and row["condition"] == condition]
        mean, low, high = confidence([row["accuracy"] for row in selected], seed=len(aggregate))
        aggregate.append({"variant": variant, "condition": condition, "mean_accuracy": mean, "seed_ci_low": low, "seed_ci_high": high, "mean_loss": float(np.mean([row["loss"] for row in selected])), "mean_ms_per_example": float(np.mean([row.get("milliseconds_per_example", float("nan")) for row in selected]))})
    write_summary(output / "condition_summary.csv", aggregate)
    baseline = [row["accuracy"] for row in condition_rows if row["variant"] == "matched_baseline"]
    gated = [row["accuracy"] for row in condition_rows if row["variant"] == "gated" and row["condition"] == "native"]
    forced_open = [row["accuracy"] for row in condition_rows if row["variant"] == "gated" and row["condition"] == "forced_open"]
    effect = [right - left for left, right in zip(baseline, gated)]
    effect_mean, effect_low, effect_high = confidence(effect, seed=303)
    native_open_effect = [native - opened for native, opened in zip(gated, forced_open)]
    native_open_mean, native_open_low, native_open_high = confidence(native_open_effect, seed=304)
    gates = np.asarray([row["gate"] for row in all_gate_records])
    utility = np.asarray([row["causal_utility"] for row in all_gate_records])
    magnitude = np.asarray([row["relative_update_norm"] for row in all_gate_records])
    gate_utility = spearmanr(gates, utility)
    gate_magnitude = spearmanr(gates, magnitude)
    harmful = [row["gate"] for row in all_gate_records if row["e1_candidate_harmful_cell"]]
    other = [row["gate"] for row in all_gate_records if not row["e1_candidate_harmful_cell"]]
    overall = {
        "stage": "E3", "seeds": len(base["training"]["seeds"]), "parameter_counts": parameter_counts,
        "matched_parameter_count": len(set(parameter_counts.values())) == 1,
        "gated_minus_baseline_accuracy": {"mean": effect_mean, "seed_ci_low": effect_low, "seed_ci_high": effect_high},
        "native_minus_forced_open_accuracy": {"mean": native_open_mean, "seed_ci_low": native_open_low, "seed_ci_high": native_open_high},
        "mean_gate": float(gates.mean()), "mean_gate_entropy": float(np.mean([row["gate_entropy"] for row in all_gate_records])),
        "median_gate": float(np.median(gates)), "fraction_gate_below_0_9": float(np.mean(gates < 0.9)),
        "mean_gate_e1_candidate_harmful_cell": float(np.mean(harmful)), "mean_gate_other_cells": float(np.mean(other)),
        "gate_utility_spearman": {"rho": float(gate_utility.statistic), "p": float(gate_utility.pvalue)},
        "gate_relative_magnitude_spearman": {"rho": float(gate_magnitude.statistic), "p": float(gate_magnitude.pvalue)},
        "soft_active_block_fraction": 1.0, "soft_gating_saves_flops": False,
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
