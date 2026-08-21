"""Run E2 same-content/different-goal counterfactual analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, load_seed, make_loader, resolve_device, write_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e2_counterfactual.yaml")
    parser.add_argument("--output", default="results/e2")
    return parser.parse_args()


def read_e1_records(path: Path) -> dict[tuple[int, str, int], dict]:
    records = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records[(int(row["seed"]), row["example_id"], int(row["layer"]))] = row
    return records


@torch.inference_mode()
def collect_states(model, loader, examples, device):
    model.eval()
    states = {}
    for batch in loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        output = model(ids, mask, capture=True)
        for row, example_index in enumerate(batch["example_index"].tolist()):
            token_index = int(mask[row].sum()) - 1
            states[examples[example_index].example_id] = [
                layer_state[row, token_index].float().cpu() for layer_state in output.states[1:]
            ]
    return states


def finite_correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {"pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue), "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}


def main():
    args = parse_args()
    config = read_yaml(args.config)
    e1_config = read_yaml(config["source_config"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    e1_records = read_e1_records(Path(config["source_stage"]) / "block_records.csv")
    pair_records = []

    for seed in e1_config["training"]["seeds"]:
        checkpoint = Path(config["source_stage"]) / "runs" / f"seed-{seed}" / "model.pt"
        model, splits, vocabulary, datasets = load_seed(e1_config, checkpoint)
        device = resolve_device(e1_config["training"]["device"])
        model.to(device)
        loader = make_loader(datasets["test"], e1_config, shuffle=False, seed=int(seed))
        states = collect_states(model, loader, splits["test"], device)
        families = defaultdict(list)
        for example in splits["test"]:
            families[example.family_id].append(example)
        seed_rows = []
        for family_examples in families.values():
            for left, right in itertools.combinations(sorted(family_examples, key=lambda item: item.intent), 2):
                for layer in range(model.num_layers):
                    left_record = e1_records[(int(seed), left.example_id, layer)]
                    right_record = e1_records[(int(seed), right.example_id, layer)]
                    left_state, right_state = states[left.example_id][layer], states[right.example_id][layer]
                    delta = left_state - right_state
                    record = {
                        "seed": seed, "family_id": left.family_id, "layer": layer,
                        "intent_a": left.intent, "intent_b": right.intent,
                        "task_pair": f"{left.intent}__{right.intent}",
                        "both_learned_tasks": {left.intent, right.intent} <= {"maximum", "minimum"},
                        "identifiability_a": left.goal_identifiability,
                        "identifiability_b": right.goal_identifiability,
                        "identifiability_delta": left.goal_identifiability - right.goal_identifiability,
                        "residual_delta_norm": float(torch.linalg.vector_norm(delta)),
                        "residual_cosine": float(torch.nn.functional.cosine_similarity(left_state, right_state, dim=0)),
                        "utility_a": float(left_record["utility"]), "utility_b": float(right_record["utility"]),
                        "utility_delta": float(left_record["utility"]) - float(right_record["utility"]),
                        "absolute_utility_delta": abs(float(left_record["utility"]) - float(right_record["utility"])),
                        "target_probability_delta": float(left_record["full_probability"]) - float(right_record["full_probability"]),
                        "attention_entropy_delta": float(left_record["attention_entropy"]) - float(right_record["attention_entropy"]),
                    }
                    pair_records.append(record)
                    seed_rows.append(record)
        run_dir = output / "runs" / f"seed-{seed}"
        write_summary(run_dir / "pair_records.csv", seed_rows)
        metadata = RunMetadata.collect(
            run_id=f"e2-seed-{seed}", model="tiny_residual_decoder", model_variant="baseline",
            dataset="synthetic_counterfactual_v1", seed=int(seed), context_length=int(e1_config["data"]["max_length"]),
            batch_size=int(e1_config["training"]["batch_size"]), dtype="float32", device=device,
        )
        save_manifest(run_dir / "manifest.json", metadata, {"e2": config, "e1": e1_config})

    write_summary(output / "pair_records.csv", pair_records)
    summaries = []
    minimum = float(config["statistics"]["minimum_utility_delta"])
    for task_pair in sorted({row["task_pair"] for row in pair_records}):
        for layer in range(int(e1_config["model"]["layers"])):
            selected = [row for row in pair_records if row["task_pair"] == task_pair and row["layer"] == layer]
            seed_means = []
            for seed in e1_config["training"]["seeds"]:
                values = [row["utility_delta"] for row in selected if row["seed"] == seed]
                seed_means.append(float(np.mean(values)))
            mean, low, high = confidence(seed_means, seed=100 + layer)
            positive = sum(value > minimum for value in seed_means)
            negative = sum(value < -minimum for value in seed_means)
            replicated = (low > minimum and positive >= 4) or (high < -minimum and negative >= 4)
            summaries.append({
                "task_pair": task_pair, "layer": layer, "mean_utility_delta": mean,
                "seed_ci_low": low, "seed_ci_high": high, "positive_seed_count": positive,
                "negative_seed_count": negative, "replicated_goal_effect": replicated,
                "mean_residual_delta_norm": float(np.mean([row["residual_delta_norm"] for row in selected])),
                "mean_residual_cosine": float(np.mean([row["residual_cosine"] for row in selected])),
            })
    write_summary(output / "pair_layer_summary.csv", summaries)

    individual = []
    for row in e1_records.values():
        individual.append((float(row["goal_identifiability"]), float(row["utility"])))
    correlations = finite_correlation([item[0] for item in individual], [item[1] for item in individual])
    learned_summaries = [row for row in summaries if row["task_pair"] == "maximum__minimum"]
    overall = {
        "stage": "E2", "seeds": len(e1_config["training"]["seeds"]),
        "content_families_per_seed": int(e1_config["data"]["test_families"]),
        "counterfactual_pairs": len(pair_records),
        "replicated_all_task_pair_layer_effects": sum(row["replicated_goal_effect"] for row in summaries),
        "replicated_learned_only_effects": sum(row["replicated_goal_effect"] for row in learned_summaries),
        "goal_identifiability_utility_correlation": correlations,
        "h1_supported_all_tasks": any(row["replicated_goal_effect"] for row in summaries),
        "h1_supported_learned_tasks": any(row["replicated_goal_effect"] for row in learned_summaries),
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
