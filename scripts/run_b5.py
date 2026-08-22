"""Run Cycle B5 expanded competence-matched task ecology."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from gated_residuals.artifacts import RunMetadata, save_manifest, save_records_csv, save_records_parquet
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, current_commit, load_seed, make_loader, resolve_device, train_seed
from gated_residuals.standard_metrics import target_log_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b5_task_ecology.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def finite(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite B5 metric")
    return result


def last_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = mask.sum(1).sub(1)
    return values[torch.arange(values.size(0), device=values.device), indices]


def token_bins(values: torch.Tensor, length: int, bins: int) -> torch.Tensor:
    output = []
    for token_bin in range(bins):
        start = int(token_bin * length / bins)
        stop = max(start + 1, int((token_bin + 1) * length / bins))
        output.append(values[start:stop].float().mean(0))
    return torch.stack(output)


def attention_bins(weights: torch.Tensor, length: int, bins: int) -> torch.Tensor:
    # Mean over heads for the final query, then aggregate source-token probability by relative bin.
    distribution = weights[:, length - 1, :length].float().mean(0)
    output = []
    for token_bin in range(bins):
        start = int(token_bin * length / bins)
        stop = max(start + 1, int((token_bin + 1) * length / bins))
        output.append(distribution[start:stop].sum())
    result = torch.stack(output)
    return result / result.sum().clamp_min(1e-8)


@torch.inference_mode()
def capture_validation(model, datasets, config: dict, seed: int, device: torch.device):
    loader = make_loader(datasets["val"], config, shuffle=False, seed=seed)
    features = {}
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        native = model(ids, mask, capture=True)
        for row, index in enumerate(batch["example_index"].tolist()):
            length = int(mask[row].sum())
            features[index] = torch.stack([native.states[layer][row, length - 1].cpu() for layer in range(model.num_layers)])
    return features


@torch.inference_mode()
def analyze_test(model, splits, datasets, config: dict, seed: int, device: torch.device):
    analysis_config = json.loads(json.dumps(config))
    analysis_config["training"]["batch_size"] = min(32, int(config["training"]["batch_size"]))
    loader = make_loader(datasets["test"], analysis_config, shuffle=False, seed=seed)
    bins = int(config["capture"]["token_bins"])
    causal, quality = [], []
    features = {}
    max_parity = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); targets = batch["target"].to(device)
        plain = model(ids, mask)
        native = model(ids, mask, capture=True)
        parity = float((plain.logits - native.logits).abs().max())
        max_parity = max(max_parity, parity)
        if not torch.equal(plain.logits, native.logits):
            raise RuntimeError(f"B5 native/capture parity failed: {parity}")
        full_lp = target_log_probability(native.logits, targets).cpu()
        predictions = native.logits.argmax(-1).cpu()
        interventions = []
        for layer in range(model.num_layers):
            sa = model(ids, mask, skip_attention_layers={layer})
            ff = model(ids, mask, skip_ff_layers={layer})
            block = model(ids, mask, skip_layers={layer})
            interventions.append((target_log_probability(sa.logits, targets).cpu(), target_log_probability(ff.logits, targets).cpu(), target_log_probability(block.logits, targets).cpu()))
        for row, index in enumerate(batch["example_index"].tolist()):
            example = splits["test"][index]
            length = int(mask[row].sum())
            correct = bool(predictions[row] == targets[row].cpu())
            quality.append({"seed": seed, "task_family": example.intent, "family_id": example.family_id, "example_id": example.example_id, "correct": correct, "target_logprob": finite(full_lp[row])})
            state_maps, sa_maps, ff_maps, attention_maps = [], [], [], []
            for layer in range(model.num_layers):
                sa_lp, ff_lp, block_lp = interventions[layer]
                token = length - 1
                attention_update = native.attention_candidates[layer][row, token].float()
                ff_update = native.ff_candidates[layer][row, token].float()
                causal.append({
                    "cycle": "B", "experiment": "B5", "revision": int(config["revision"]), "model_family": "tiny_custom_transformer", "model_variant": "baseline",
                    "depth": model.num_layers, "width": model.width, "task_family": example.intent, "family_id": example.family_id, "example_id": example.example_id,
                    "seed": seed, "layer": layer, "full_correct": correct, "full_target_logprob": finite(full_lp[row]),
                    "utility_block": finite(full_lp[row] - block_lp[row]), "utility_sa": finite(full_lp[row] - sa_lp[row]), "utility_ff": finite(full_lp[row] - ff_lp[row]),
                    "sa_update_norm": finite(torch.linalg.vector_norm(attention_update)), "ff_update_norm": finite(torch.linalg.vector_norm(ff_update)),
                    "sa_ff_cosine": finite(F.cosine_similarity(attention_update[None], ff_update[None])[0]),
                    "attention_entropy": finite(-(native.attention[layer][row, :, token, :length].float().clamp_min(1e-8) * native.attention[layer][row, :, token, :length].float().clamp_min(1e-8).log()).sum(-1).mean()),
                })
                state_maps.append(token_bins(native.states[layer][row, :length].cpu(), length, bins))
                sa_maps.append(token_bins(native.attention_candidates[layer][row, :length].cpu(), length, bins))
                ff_maps.append(token_bins(native.ff_candidates[layer][row, :length].cpu(), length, bins))
                attention_maps.append(attention_bins(native.attention[layer][row].cpu(), length, bins))
            features[index] = {"state": torch.stack(state_maps), "sa": torch.stack(sa_maps), "ff": torch.stack(ff_maps), "attention": torch.stack(attention_maps)}
    return causal, quality, features, max_parity


def goal_probe(val_features: dict, test_features: dict, val_examples, test_examples, depth: int, seed: int):
    tasks = sorted({example.intent for example in val_examples})
    task_index = {task: index for index, task in enumerate(tasks)}
    rows = []
    for layer in range(depth):
        train_x = torch.stack([val_features[index][layer] for index in range(len(val_examples))]).float()
        test_x = torch.stack([test_features[index]["state"][layer, -1] for index in range(len(test_examples))]).float()
        mean = train_x.mean(0); scale = train_x.std(0).clamp_min(1e-6)
        train_x = (train_x - mean) / scale; test_x = (test_x - mean) / scale
        centroids = torch.stack([train_x[[example.intent == task for example in val_examples]].mean(0) for task in tasks])
        predicted = torch.cdist(test_x, centroids).argmin(1)
        target = torch.tensor([task_index[example.intent] for example in test_examples])
        rows.append({"seed": seed, "layer": layer, "probe": "validation_nearest_centroid_linear", "accuracy": finite((predicted == target).float().mean()), "chance_accuracy": 1 / len(tasks)})
    return rows


def goal_divergence(features: dict, examples, config: dict, seed: int):
    bins = int(config["capture"]["token_bins"]); depth = int(config["model"]["layers"])
    by_family = defaultdict(dict)
    for index, example in enumerate(examples):
        by_family[example.family_id][example.intent] = features[index]
    accum = defaultdict(list)
    tasks = list(config["data"]["intents"])
    for family in by_family.values():
        for left_task, right_task in itertools.combinations(tasks, 2):
            left, right = family[left_task], family[right_task]
            pair = f"{left_task}__{right_task}"
            for layer in range(depth):
                for token_bin in range(bins):
                    record = {"seed": seed, "task_pair": pair, "layer": layer, "token_bin": token_bin}
                    for component in ("state", "sa", "ff"):
                        a, b = left[component][layer, token_bin].float(), right[component][layer, token_bin].float()
                        record[f"{component}_distance"] = finite(torch.linalg.vector_norm(a - b))
                        record[f"{component}_cosine"] = finite(F.cosine_similarity(a[None], b[None])[0])
                    p = left["attention"][layer].clamp_min(1e-8); q = right["attention"][layer].clamp_min(1e-8); midpoint = (p + q) / 2
                    record["attention_js_divergence"] = finite(0.5 * ((p * (p / midpoint).log()).sum() + (q * (q / midpoint).log()).sum()))
                    accum[(pair, layer, token_bin)].append(record)
    rows = []
    for (pair, layer, token_bin), values in accum.items():
        row = {"seed": seed, "task_pair": pair, "layer": layer, "token_bin": token_bin, "family_n": len(values)}
        for metric in ("state_distance", "state_cosine", "sa_distance", "sa_cosine", "ff_distance", "ff_cosine", "attention_js_divergence"):
            row[metric] = sum(item[metric] for item in values) / len(values)
        rows.append(row)
    return rows


def summarize(causal: list[dict], quality: list[dict], config: dict):
    frame, qframe = pd.DataFrame(causal), pd.DataFrame(quality)
    tasks = list(config["data"]["intents"]); seeds = list(config["training"]["seeds"])
    minimum_seeds = math.ceil(float(config["statistics"]["minimum_seed_fraction"]) * len(seeds))
    seed_quality = qframe.groupby(["task_family", "seed"], as_index=False).agg(accuracy=("correct", "mean"), target_logprob=("target_logprob", "mean"))
    task_rows = []
    for task, group in seed_quality.groupby("task_family"):
        mean, low, high = confidence(group.accuracy.tolist(), seed=len(task))
        task_rows.append({"task_family": task, "seed_n": len(group), "accuracy_mean": mean, "accuracy_ci95_low": low, "accuracy_ci95_high": high, "competent_seed_count": int((group.accuracy >= float(config["statistics"]["competence_accuracy"])).sum()), "competent": bool((group.accuracy >= float(config["statistics"]["competence_accuracy"])).sum() >= minimum_seeds)})
    task_frame = pd.DataFrame(task_rows)
    pair_rows = []
    delta = float(config["statistics"]["competence_delta"])
    for left, right in itertools.combinations(tasks, 2):
        l = seed_quality[seed_quality.task_family == left].set_index("seed").accuracy
        r = seed_quality[seed_quality.task_family == right].set_index("seed").accuracy
        differences = (l - r).abs()
        competent = bool(task_frame.set_index("task_family").loc[left, "competent"] and task_frame.set_index("task_family").loc[right, "competent"] and int((differences < delta).sum()) >= minimum_seeds and abs(float(l.mean() - r.mean())) < delta)
        pair_rows.append({"task_left": left, "task_right": right, "seed_n": len(differences), "accuracy_left": float(l.mean()), "accuracy_right": float(r.mean()), "mean_absolute_seed_difference": float(differences.mean()), "within_delta_seed_count": int((differences < delta).sum()), "competence_matched": competent})
    pair_frame = pd.DataFrame(pair_rows)
    matched = {f"{row.task_left}__{row.task_right}" for row in pair_frame[pair_frame.competence_matched].itertuples()}
    indexed = frame.set_index(["seed", "family_id", "task_family", "layer"])
    contrast_seed = []
    for pair in sorted(matched):
        left, right = pair.split("__")
        for layer in range(int(config["model"]["layers"])):
            for seed in seeds:
                a = indexed.xs((seed, left, layer), level=("seed", "task_family", "layer"))
                b = indexed.xs((seed, right, layer), level=("seed", "task_family", "layer"))
                joined = a[["utility_block", "utility_sa", "utility_ff"]].join(b[["utility_block", "utility_sa", "utility_ff"]], lsuffix="_left", rsuffix="_right")
                row = {"task_pair": pair, "layer": layer, "seed": seed, "family_n": len(joined)}
                for component in ("block", "sa", "ff"):
                    row[f"utility_{component}_difference"] = float((joined[f"utility_{component}_left"] - joined[f"utility_{component}_right"]).mean())
                contrast_seed.append(row)
    contrast_rows = []
    effect = float(config["statistics"]["utility_difference"])
    contrast_groups = pd.DataFrame(contrast_seed).groupby(["task_pair", "layer"]) if contrast_seed else []
    for (pair, layer), group in contrast_groups:
        row = {"task_pair": pair, "layer": int(layer), "seed_n": len(group)}
        for component in ("block", "sa", "ff"):
            metric = f"utility_{component}_difference"; values = group[metric]
            mean, low, high = confidence(values.tolist(), seed=int(layer) + len(component) + len(pair))
            replicated = bool(((low > effect) and int((values > effect).sum()) >= minimum_seeds) or ((high < -effect) and int((values < -effect).sum()) >= minimum_seeds))
            row[f"{metric}_mean"] = mean; row[f"{metric}_ci95_low"] = low; row[f"{metric}_ci95_high"] = high; row[f"replicated_{component}"] = replicated
        contrast_rows.append(row)
    return task_rows, pair_rows, contrast_seed, contrast_rows


def plot(task_rows, causal, output: Path):
    tasks = pd.DataFrame(task_rows).sort_values("accuracy_mean")
    frame = pd.DataFrame(causal)
    profiles = frame.groupby(["task_family", "layer"], as_index=False).utility_block.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].barh(tasks.task_family, tasks.accuracy_mean, color="#4472c4"); axes[0].axvline(.9, color="black", linestyle="--", linewidth=1); axes[0].set(xlabel="Test accuracy", xlim=(0, 1.02))
    for task, group in profiles.groupby("task_family"):
        axes[1].plot(group.layer, group.utility_block, linewidth=1.2, label=task)
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(xlabel="Layer", ylabel="Mean block utility")
    axes[1].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout(); (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures" / "b5_ecology.pdf", bbox_inches="tight"); fig.savefig(output / "figures" / "b5_ecology.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args(); config = read_yaml(args.config); output = Path(args.output or config["output"]["directory"])
    if output.exists() and args.fresh: shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = args.seed or list(config["training"]["seeds"])
    all_causal, all_quality, all_goal_maps, all_probes, registry = [], [], [], [], []
    started = time.perf_counter()
    for seed in seeds:
        run_dir = output / "runs" / f"seed-{seed}"; checkpoint = run_dir / "model.pt"
        if checkpoint.exists():
            model, splits, vocabulary, datasets = load_seed(config, checkpoint); training = {"training_seconds": 0.0, "best_val_loss": min(row["val_loss"] for row in json.loads((run_dir / "history.json").read_text(encoding="utf-8")))}
        else:
            model, splits, vocabulary, datasets, training = train_seed(config, "baseline", int(seed), run_dir)
        device_name = resolve_device(config["training"].get("device", "auto")); device = torch.device(device_name); model.to(device).eval()
        val_features = capture_validation(model, datasets, config, int(seed), device)
        causal, quality, test_features, parity = analyze_test(model, splits, datasets, config, int(seed), device)
        probes = goal_probe(val_features, test_features, splits["val"], splits["test"], model.num_layers, int(seed))
        maps = goal_divergence(test_features, splits["test"], config, int(seed))
        all_causal.extend(causal); all_quality.extend(quality); all_probes.extend(probes); all_goal_maps.extend(maps)
        registry.append({"cycle": "B", "experiment": "B5", "revision": int(config["revision"]), "run_id": f"depth16-width64-seed-{seed}", "model_family": "tiny_custom_transformer", "model_variant": "baseline", "depth": model.num_layers, "width": model.width, "task_family": "+".join(config["data"]["intents"]), "seed": int(seed), "checkpoint": str(checkpoint).replace("\\", "/"), "parameters": sum(p.numel() for p in model.parameters()), "test_examples": len(splits["test"]), "native_capture_max_logit_error": parity, "best_val_loss": training["best_val_loss"], "training_seconds": training["training_seconds"], "device": device_name})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if set(seeds) != set(config["training"]["seeds"]):
        print(json.dumps({"status": "partial", "seeds": seeds, "runs": len(registry)}, indent=2)); return
    task_rows, pair_rows, contrast_seed, contrast_rows = summarize(all_causal, all_quality, config)
    save_records_parquet(output / "causal_records.parquet", all_causal); save_records_parquet(output / "quality_records.parquet", all_quality)
    save_records_parquet(output / "goal_divergence_maps.parquet", all_goal_maps); save_records_csv(output / "goal_probe.csv", all_probes)
    save_records_csv(output / "run_registry.csv", registry); save_records_csv(output / "task_competence.csv", task_rows); save_records_csv(output / "competence_pairs.csv", pair_rows)
    save_records_csv(output / "utility_contrast_seed.csv", contrast_seed or [{"status": "no_competence_matched_pairs"}]); save_records_csv(output / "utility_contrasts.csv", contrast_rows or [{"status": "no_competence_matched_pairs"}])
    plot(task_rows, all_causal, output)
    contrast_frame = pd.DataFrame(contrast_rows)
    replicated = int(contrast_frame[["replicated_block", "replicated_sa", "replicated_ff"]].sum().sum()) if contrast_rows else 0
    summary = {"cycle": "B", "experiment": "B5", "revision": int(config["revision"]), "source_commit": current_commit(), "runs": len(registry), "tasks": len(task_rows), "causal_rows": len(all_causal), "quality_rows": len(all_quality), "goal_map_rows": len(all_goal_maps), "native_capture_max_logit_error": max(row["native_capture_max_logit_error"] for row in registry), "competent_tasks": [row["task_family"] for row in task_rows if row["competent"]], "competence_matched_pairs": sum(row["competence_matched"] for row in pair_rows), "replicated_task_conditioned_utility_cells": replicated, "goal_probe_final_layer_accuracy": float(pd.DataFrame(all_probes).query(f"layer == {int(config['model']['layers']) - 1}").accuracy.mean()), "b6_gate_open_from_b5": bool(replicated > 0), "elapsed_seconds": time.perf_counter() - started, "interpretation": "Only competence-matched pairs enter task-conditioned utility contrasts; goal divergence and attention redistribution remain descriptive."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text("# B5 failure and null-result notes\n\n- Aggregate tasks remain excluded because competence was not established.\n- A one-seed validation pilot excluded argmax-position (0.567) and argmin-position (0.583).\n- A five-seed mixed-identifiability pilot left task validation means at 0.813--0.917 and no competence-matched pair; B5 revision 3 therefore uses high-identifiability wording only. Pilot frozen-test measurements are not retained or used in claims.\n- Only preregistered competence-matched pairs enter utility claims.\n- Goal maps compare relative token bins because natural-language instructions have unequal lengths.\n- Nearest-centroid goal probes are validation-fit linear discriminants, not causal controls.\n", encoding="utf-8")
    metadata = RunMetadata.collect(run_id="cycle-b-b5", model="tiny_residual_decoder", model_variant="baseline_depth16", dataset="synthetic_counterfactual_v2_selection", seed=0, context_length=int(config["data"]["max_length"]), batch_size=int(config["training"]["batch_size"]), dtype="float32", device=registry[0]["device"])
    save_manifest(output / "manifest.json", metadata, config); (output / "run.log").write_text(f"B5 complete runs={len(registry)} elapsed_seconds={summary['elapsed_seconds']:.3f}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
