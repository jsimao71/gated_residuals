"""Run E4 shared-goal-latent training and future-utility probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, evaluate, load_seed, make_loader, resolve_device, train_seed, write_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e4_goal_latent.yaml")
    parser.add_argument("--output", default="results/e4")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def collect_probe_data(model, loader, examples, device):
    rows = []
    model.eval()
    for batch in loader:
        ids, mask, target = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
        native = model(ids, mask, capture=True)
        full_probability = native.logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
        skipped_probability = [model(ids, mask, skip_layers={layer}).logits.softmax(-1).gather(1, target[:, None]).squeeze(1) for layer in range(model.num_layers)]
        for row_index, example_index in enumerate(batch["example_index"].tolist()):
            example = examples[example_index]
            token_index = int(mask[row_index].sum()) - 1
            rows.append({
                "example_id": example.example_id, "task": example.intent,
                "goal": [state[row_index].float().cpu().numpy() for state in native.goal_states],
                "residual": [state[row_index, token_index].float().cpu().numpy() for state in native.states[1:]],
                "pooled": [state[row_index, : token_index + 1].float().mean(0).cpu().numpy() for state in native.states[1:]],
                "utility": [float(full_probability[row_index] - probability[row_index]) for probability in skipped_probability],
            })
    return rows


def ridge_predict(train_x, train_y, test_x, alpha):
    train_x, test_x = np.asarray(train_x, dtype=np.float64), np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weights


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    residual = np.square(target - prediction).sum()
    total = np.square(target - target.mean()).sum()
    return {"r2": float(1 - residual / total) if total > 1e-12 else float("nan"), "mae": float(np.abs(target - prediction).mean())}


def control_features(name, rows, source_layer, rng, reference=None):
    if name == "goal_latent":
        return np.stack([row["goal"][source_layer] for row in rows])
    if name == "current_residual":
        return np.stack([row["residual"][source_layer] for row in rows])
    if name == "pooled_residual":
        return np.stack([row["pooled"][source_layer] for row in rows])
    goals = np.stack([row["goal"][source_layer] for row in rows])
    if name == "shuffled_goal":
        return goals[rng.permutation(len(goals))]
    if name == "moment_random":
        if reference is None:
            mean, std = goals.mean(0), goals.std(0)
        else:
            mean, std = reference
        return rng.normal(size=goals.shape) * std + mean
    raise ValueError(name)


@torch.inference_mode()
def goal_intervention_quality(model, loader, device):
    return {mode: evaluate(model, loader, device, goal_mode=mode)["accuracy"] for mode in ("native", "shuffled", "zero")}


def main():
    args = parse_args()
    config = read_yaml(args.config)
    base = read_yaml(config["base_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    probe_rows, quality_rows = [], []
    parameter_counts = {}
    for variant in config["variants"]:
        for seed in base["training"]["seeds"]:
            run_dir = output / "runs" / variant / f"seed-{seed}"
            if args.reuse_checkpoints and (run_dir / "model.pt").exists():
                model, splits, vocabulary, datasets = load_seed(base, run_dir / "model.pt")
            else:
                model, splits, vocabulary, datasets, _ = train_seed(base, variant, int(seed), run_dir)
            device = resolve_device(base["training"]["device"])
            model.to(device)
            parameter_counts[variant] = sum(parameter.numel() for parameter in model.parameters())
            test_loader = make_loader(datasets["test"], base, shuffle=False, seed=int(seed))
            quality = goal_intervention_quality(model, test_loader, device)
            quality_rows.append({"variant": variant, "seed": seed, **{f"{mode}_accuracy": value for mode, value in quality.items()}})
            if variant == "goal":
                val_loader = make_loader(datasets["val"], base, shuffle=False, seed=int(seed))
                train_rows = collect_probe_data(model, val_loader, splits["val"], device)
                test_rows = collect_probe_data(model, test_loader, splits["test"], device)
                rng = np.random.default_rng(int(seed) + 404)
                for source_layer in range(model.num_layers - 1):
                    train_goals = np.stack([row["goal"][source_layer] for row in train_rows])
                    moments = (train_goals.mean(0), train_goals.std(0))
                    for target_layer in range(source_layer + 1, model.num_layers):
                        train_y = [row["utility"][target_layer] for row in train_rows]
                        test_y = [row["utility"][target_layer] for row in test_rows]
                        for control in config["probe"]["controls"]:
                            train_x = control_features(control, train_rows, source_layer, rng, moments)
                            test_x = control_features(control, test_rows, source_layer, rng, moments)
                            prediction = ridge_predict(train_x, train_y, test_x, float(config["probe"]["ridge_alpha"]))
                            probe_rows.append({
                                "seed": seed, "source_layer": source_layer, "target_layer": target_layer,
                                "control": control, **metrics(test_y, prediction),
                            })
            metadata = RunMetadata.collect(
                run_id=f"e4-{variant}-seed-{seed}", model="tiny_residual_decoder", model_variant=variant,
                dataset="synthetic_counterfactual_v1", seed=int(seed), context_length=int(base["data"]["max_length"]),
                batch_size=int(base["training"]["batch_size"]), dtype="float32", device=device,
            )
            save_manifest(run_dir / "e4_manifest.json", metadata, {"e4": config, "base": base})
    write_summary(output / "probe_results.csv", probe_rows)
    write_summary(output / "quality_by_seed.csv", quality_rows)
    aggregate = []
    for control in config["probe"]["controls"]:
        selected = [row for row in probe_rows if row["control"] == control]
        aggregate.append({"control": control, "mean_test_r2": float(np.nanmean([row["r2"] for row in selected])), "median_test_r2": float(np.nanmedian([row["r2"] for row in selected])), "mean_test_mae": float(np.mean([row["mae"] for row in selected])), "positive_r2_fraction": float(np.mean([row["r2"] > 0 for row in selected]))})
    write_summary(output / "probe_summary.csv", aggregate)
    goal_r2 = {(row["seed"], row["source_layer"], row["target_layer"]): row["r2"] for row in probe_rows if row["control"] == "goal_latent"}
    comparisons = {}
    for control in ("current_residual", "pooled_residual", "moment_random", "shuffled_goal"):
        other = {(row["seed"], row["source_layer"], row["target_layer"]): row["r2"] for row in probe_rows if row["control"] == control}
        differences = [goal_r2[key] - other[key] for key in goal_r2]
        comparisons[control] = {"mean_r2_advantage": float(np.mean(differences)), "win_fraction": float(np.mean(np.asarray(differences) > float(config["statistics"]["minimum_r2_advantage"])))}
    goal_quality = [row for row in quality_rows if row["variant"] == "goal"]
    goal_only_quality = [row for row in quality_rows if row["variant"] == "goal_only"]
    overall = {
        "stage": "E4", "seeds": len(base["training"]["seeds"]), "parameter_counts": parameter_counts,
        "mean_goal_model_accuracy": float(np.mean([row["native_accuracy"] for row in goal_quality])),
        "mean_goal_only_accuracy": float(np.mean([row["native_accuracy"] for row in goal_only_quality])),
        "mean_goal_shuffled_accuracy": float(np.mean([row["shuffled_accuracy"] for row in goal_quality])),
        "mean_goal_zeroed_accuracy": float(np.mean([row["zero_accuracy"] for row in goal_quality])),
        "probe_summary": {row["control"]: row for row in aggregate}, "goal_latent_comparisons": comparisons,
        "shared_goal_control_supported": comparisons["current_residual"]["mean_r2_advantage"] > float(config["statistics"]["minimum_r2_advantage"]) and comparisons["current_residual"]["win_fraction"] >= float(config["statistics"]["minimum_seed_fraction"]),
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
