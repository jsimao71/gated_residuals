"""Run Cycle B4: controlled depth/capacity sweep with causal residual probes."""

from __future__ import annotations

import argparse
import copy
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
from gated_residuals.residual_dynamics import amplification_repair, pairwise_layer_matrices, sa_ff_geometry
from gated_residuals.standard_metrics import target_log_probability
from gated_residuals.temporal_stability import autocorrelation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b4_depth_sweep.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def finite(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite B4 metric")
    return result


def condition_config(config: dict, condition: dict) -> dict:
    result = copy.deepcopy(config)
    result["model"]["layers"] = int(condition["depth"])
    result["model"]["width"] = int(condition["width"])
    result.pop("conditions", None)
    result.pop("output", None)
    return result


def last_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = mask.sum(1).sub(1)
    return values[torch.arange(values.size(0), device=values.device), indices]


def intermediate_logprob(model, state: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return target_log_probability(model.output(model.final_norm(last_token(state, mask))), targets)


@torch.inference_mode()
def analyze(model, splits, datasets, run_config: dict, condition: dict, seed: int):
    device_name = resolve_device(run_config["training"].get("device", "auto"))
    device = torch.device(device_name)
    model.to(device).eval()
    analysis_config = copy.deepcopy(run_config)
    analysis_config["training"]["batch_size"] = min(32, int(run_config["training"]["batch_size"]))
    loader = make_loader(datasets["test"], analysis_config, shuffle=False, seed=seed)
    rows: list[dict] = []
    quality_rows: list[dict] = []
    final_updates = defaultdict(lambda: [[] for _ in range(model.num_layers)])
    final_states = defaultdict(lambda: [[] for _ in range(model.num_layers)])
    max_parity = 0.0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        plain = model(ids, mask)
        native = model(ids, mask, capture=True)
        parity = float((plain.logits - native.logits).abs().max())
        max_parity = max(max_parity, parity)
        if not torch.equal(plain.logits, native.logits):
            raise RuntimeError(f"B4 native/capture parity failed: {parity}")
        full_lp = target_log_probability(native.logits, targets)
        full_lp_cpu = full_lp.cpu()
        predictions = native.logits.argmax(-1)

        intervention = []
        for layer in range(model.num_layers):
            skip_sa = model(ids, mask, skip_attention_layers={layer})
            skip_ff = model(ids, mask, skip_ff_layers={layer})
            skip_block = model(ids, mask, skip_layers={layer}, capture=True)
            trajectories = []
            task_gaps = []
            for state_index in range(layer + 1, model.num_layers + 1):
                native_state = native.states[state_index]
                skipped_state = skip_block.states[state_index]
                trajectories.append(torch.linalg.vector_norm((last_token(native_state, mask) - last_token(skipped_state, mask)).float(), dim=-1))
                native_proxy = intermediate_logprob(model, native_state, mask, targets)
                skipped_proxy = intermediate_logprob(model, skipped_state, mask, targets)
                task_gaps.append((native_proxy - skipped_proxy).abs())
            repair_flags = []
            for row_index in range(ids.size(0)):
                vector_repair = False
                task_repair = False
                if len(trajectories) >= 3:
                    vector = amplification_repair(torch.stack(trajectories)[:, row_index], amplification_ratio=float(run_config["statistics"]["amplification_ratio"]), repair_fraction=float(run_config["statistics"]["repair_fraction"]))
                    task = amplification_repair(torch.stack(task_gaps)[:, row_index], amplification_ratio=float(run_config["statistics"]["amplification_ratio"]), repair_fraction=float(run_config["statistics"]["repair_fraction"]))
                    vector_repair = bool(vector["detected"])
                    task_repair = bool(task["detected"])
                repair_flags.append((vector_repair, task_repair, vector_repair and task_repair))
            intervention.append((
                target_log_probability(skip_sa.logits, targets).cpu(),
                target_log_probability(skip_ff.logits, targets).cpu(),
                target_log_probability(skip_block.logits, targets).cpu(),
                repair_flags,
            ))

        geometry = [sa_ff_geometry(native.states[l], native.attention_candidates[l], native.ff_candidates[l]) for l in range(model.num_layers)]
        for row_index, example_index in enumerate(batch["example_index"].tolist()):
            example = splits["test"][example_index]
            length = int(mask[row_index].sum())
            quality_rows.append({
                "condition": condition["name"], "depth": model.num_layers, "width": model.width,
                "matched_parameter_control": bool(condition["matched_parameter_control"]),
                "seed": seed, "task_family": example.intent, "example_id": example.example_id,
                "correct": bool(predictions[row_index] == targets[row_index]),
                "target_logprob": finite(full_lp[row_index]),
            })
            for layer in range(model.num_layers):
                sa_lp, ff_lp, block_lp, repair_flags = intervention[layer]
                residual_norms = torch.linalg.vector_norm(native.states[layer][row_index, :length].float(), dim=-1)
                update_norms = torch.linalg.vector_norm(native.candidates[layer][row_index, :length].float(), dim=-1)
                residual_acf = autocorrelation(residual_norms[:, None], min(1, length - 1), token_dim=0)[-1]
                update_acf = autocorrelation(update_norms[:, None], min(1, length - 1), token_dim=0)[-1]
                vector_repair, task_repair, repair_event = repair_flags[row_index]
                token = length - 1
                values = geometry[layer]
                rows.append({
                    "cycle": "B", "experiment": "B4", "revision": int(run_config["revision"]),
                    "condition": condition["name"], "model_family": "tiny_custom_transformer",
                    "model_variant": "baseline", "depth": model.num_layers, "width": model.width,
                    "matched_parameter_control": bool(condition["matched_parameter_control"]),
                    "task_family": example.intent, "example_id": example.example_id, "seed": seed, "layer": layer,
                    "full_correct": bool(predictions[row_index] == targets[row_index]),
                    "full_target_logprob": finite(full_lp_cpu[row_index]),
                    "utility_block": finite(full_lp_cpu[row_index] - block_lp[row_index]),
                    "utility_sa": finite(full_lp_cpu[row_index] - sa_lp[row_index]),
                    "utility_ff": finite(full_lp_cpu[row_index] - ff_lp[row_index]),
                    "block_update_norm": finite(torch.linalg.vector_norm(native.candidates[layer][row_index, token].float())),
                    "sa_update_norm": finite(values["attention_update_norm"][row_index, token]),
                    "ff_update_norm": finite(values["ff_update_norm"][row_index, token]),
                    "sa_ff_cosine": finite(values["attention_ff_cosine"][row_index, token]),
                    "sa_ff_cancellation": finite(values["attention_ff_cancellation"][row_index, token]),
                    "residual_norm_acf_lag1": finite(residual_acf), "update_norm_acf_lag1": finite(update_acf),
                    "vector_repair": vector_repair, "task_repair": task_repair, "repair_event": repair_event,
                })
                final_states[example.intent][layer].append(native.states[layer][row_index, token].cpu())
                final_updates[example.intent][layer].append(native.candidates[layer][row_index, token].cpu())

    matrix_rows = []
    for task in sorted(final_updates):
        states = torch.stack([torch.stack(items) for items in final_states[task]])
        updates = torch.stack([torch.stack(items) for items in final_updates[task]])
        matrices = pairwise_layer_matrices(states, updates)
        for matrix_name, matrix in matrices.items():
            for left in range(model.num_layers):
                for right in range(model.num_layers):
                    matrix_rows.append({
                        "condition": condition["name"], "depth": model.num_layers, "width": model.width,
                        "seed": seed, "task_family": task, "matrix": matrix_name,
                        "layer_i": left, "layer_j": right, "value": finite(matrix[left, right]),
                    })
    return rows, quality_rows, matrix_rows, max_parity, device_name


def summarize(records: list[dict], quality: list[dict], parameters: dict[str, int], config: dict):
    frame = pd.DataFrame(records)
    qframe = pd.DataFrame(quality)
    eps = float(config["statistics"]["negligible_utility"])
    seed_rows = []
    for keys, group in frame.groupby(["condition", "depth", "width", "matched_parameter_control", "task_family", "seed"]):
        condition, depth, width, matched, task, seed = keys
        row = {"condition": condition, "depth": int(depth), "width": int(width), "matched_parameter_control": bool(matched), "task_family": task, "seed": int(seed)}
        q = qframe[(qframe.condition == condition) & (qframe.task_family == task) & (qframe.seed == seed)]
        row["accuracy"] = float(q.correct.mean())
        row["mean_target_logprob"] = float(q.target_logprob.mean())
        for metric in ("utility_block", "utility_sa", "utility_ff", "block_update_norm", "sa_update_norm", "ff_update_norm", "residual_norm_acf_lag1", "update_norm_acf_lag1"):
            row[metric] = float(group[metric].mean())
        for component in ("block", "sa", "ff"):
            values = group[f"utility_{component}"]
            row[f"fraction_near_zero_{component}"] = float((values.abs() <= eps).mean())
            row[f"fraction_negative_{component}"] = float((values < -eps).mean())
            layer_means = group.groupby("layer")[f"utility_{component}"].mean()
            row[f"effective_{component}_count"] = int((layer_means > eps).sum())
            row[f"effective_{component}_fraction"] = row[f"effective_{component}_count"] / int(depth)
        row["repair_event_rate"] = float(group.repair_event.mean())
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    summary_rows = []
    for keys, group in seed_frame.groupby(["condition", "depth", "width", "matched_parameter_control", "task_family"]):
        condition, depth, width, matched, task = keys
        row = {"condition": condition, "depth": int(depth), "width": int(width), "parameters": parameters[condition], "matched_parameter_control": bool(matched), "task_family": task, "seed_n": len(group)}
        for metric in [column for column in seed_frame.columns if column not in {"condition", "depth", "width", "matched_parameter_control", "task_family", "seed"}]:
            values = group[metric].tolist()
            mean, low, high = confidence(values, seed=int(depth) + len(metric) + len(task))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    return seed_rows, summary_rows


def summarize_layers(records: list[dict], config: dict) -> list[dict]:
    frame = pd.DataFrame(records)
    eps = float(config["statistics"]["negligible_utility"])
    minimum_seeds = math.ceil(float(config["statistics"]["minimum_seed_fraction"]) * len(config["training"]["seeds"]))
    seed_frame = frame.groupby(["condition", "depth", "width", "matched_parameter_control", "task_family", "layer", "seed"], as_index=False)[["utility_block", "utility_sa", "utility_ff", "repair_event"]].mean()
    rows = []
    for keys, group in seed_frame.groupby(["condition", "depth", "width", "matched_parameter_control", "task_family", "layer"]):
        condition, depth, width, matched, task, layer = keys
        row = {"condition": condition, "depth": int(depth), "width": int(width), "matched_parameter_control": bool(matched), "task_family": task, "layer": int(layer), "seed_n": len(group)}
        for metric in ("utility_block", "utility_sa", "utility_ff", "repair_event"):
            values = group[metric].tolist()
            mean, low, high = confidence(values, seed=int(layer) + len(metric) + len(task))
            row[f"{metric}_mean"] = mean; row[f"{metric}_ci95_low"] = low; row[f"{metric}_ci95_high"] = high
        for component in ("block", "sa", "ff"):
            values = group[f"utility_{component}"]
            row[f"negative_{component}_seed_count"] = int((values < -eps).sum())
            row[f"near_zero_{component}_seed_count"] = int((values.abs() <= eps).sum())
            row[f"reliable_negative_{component}"] = bool(row[f"utility_{component}_ci95_high"] < -eps and row[f"negative_{component}_seed_count"] >= minimum_seeds)
            row[f"reliable_near_zero_{component}"] = bool(row[f"utility_{component}_ci95_low"] >= -eps and row[f"utility_{component}_ci95_high"] <= eps and row[f"near_zero_{component}_seed_count"] >= minimum_seeds)
        row["repeated_repair"] = bool(row["repair_event_ci95_low"] >= float(config["statistics"]["repeated_repair_rate"]))
        rows.append(row)
    return rows


def plot(summary_rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(summary_rows)
    main = frame[~frame.matched_parameter_control]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for task, group in main.groupby("task_family"):
        group = group.sort_values("depth")
        axes[0].plot(group.depth, group.accuracy_mean, marker="o", label=task)
        axes[1].plot(group.depth, group.effective_block_fraction_mean, marker="o", label=task)
    axes[0].set(xlabel="Available depth", ylabel="Test accuracy", ylim=(0, 1.03))
    axes[1].set(xlabel="Available depth", ylabel="Effective block fraction", ylim=(0, 1.03))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "depth_quality_effective_fraction.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "depth_quality_effective_fraction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    output = Path(args.output or config["output"]["directory"])
    previous_summary = {}
    if (output / "summary.json").exists():
        previous_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if output.exists() and args.fresh:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = [item for item in config["conditions"] if not args.condition or item["name"] in args.condition]
    all_records, all_quality, all_matrices, registry = [], [], [], []
    parameters = {}
    started = time.perf_counter()
    if args.postprocess_only:
        all_records = pd.read_parquet(output / "causal_records.parquet").to_dict("records")
        all_quality = pd.read_parquet(output / "quality_records.parquet").to_dict("records")
        all_matrices = [None] * len(pd.read_parquet(output / "redundancy_matrices.parquet", columns=["value"]))
        registry = pd.read_csv(output / "run_registry.csv").to_dict("records")
        parameters = {str(row["condition"]): int(row["parameters"]) for row in registry}
    for condition in ([] if args.postprocess_only else conditions):
        run_config = condition_config(config, condition)
        for seed in run_config["training"]["seeds"]:
            run_dir = output / "runs" / condition["name"] / f"seed-{seed}"
            checkpoint = run_dir / "model.pt"
            if checkpoint.exists():
                model, splits, vocabulary, datasets = load_seed(run_config, checkpoint)
                history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
                training = {"training_seconds": 0.0, "best_val_loss": min(row["val_loss"] for row in history)}
            else:
                model, splits, vocabulary, datasets, training = train_seed(run_config, "baseline", int(seed), run_dir)
            parameters[condition["name"]] = sum(parameter.numel() for parameter in model.parameters())
            records, quality, matrices, parity, device = analyze(model, splits, datasets, run_config, condition, int(seed))
            all_records.extend(records); all_quality.extend(quality); all_matrices.extend(matrices)
            registry.append({
                "cycle": "B", "experiment": "B4", "revision": int(config["revision"]),
                "run_id": f"{condition['name']}-seed-{seed}", "condition": condition["name"],
                "model_family": "tiny_custom_transformer", "model_variant": "baseline",
                "depth": int(condition["depth"]), "width": int(condition["width"]),
                "task_family": "maximum+minimum", "seed": int(seed),
                "checkpoint": str(run_dir / "model.pt").replace("\\", "/"),
                "parameters": parameters[condition["name"]], "test_examples": len(splits["test"]),
                "native_capture_max_logit_error": parity, "best_val_loss": training["best_val_loss"],
                "training_seconds": training["training_seconds"], "device": device,
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    seed_rows, summary_rows = summarize(all_records, all_quality, parameters, config)
    layer_rows = summarize_layers(all_records, config)
    if not args.postprocess_only:
        save_records_parquet(output / "causal_records.parquet", all_records)
        save_records_parquet(output / "quality_records.parquet", all_quality)
        save_records_parquet(output / "redundancy_matrices.parquet", all_matrices)
        save_records_csv(output / "run_registry.csv", registry)
    save_records_csv(output / "seed_summary.csv", seed_rows)
    save_records_csv(output / "depth_summary.csv", summary_rows)
    save_records_csv(output / "layer_summary.csv", layer_rows)
    plot(summary_rows, output)
    summary_frame = pd.DataFrame(summary_rows)
    main = summary_frame[~summary_frame.matched_parameter_control]
    competent = bool((main.accuracy_ci95_low >= float(config["statistics"]["competence_accuracy"])).all())
    fractions = main.groupby("depth").effective_block_fraction_mean.mean().sort_index()
    depth_slack = bool(competent and len(fractions) == 4 and fractions.iloc[-1] < fractions.iloc[0] - 0.10)
    layer_frame = pd.DataFrame(layer_rows)
    repeated_zero = bool(layer_frame[["reliable_near_zero_block", "reliable_near_zero_sa", "reliable_near_zero_ff"]].any(axis=None))
    repeated_negative = bool(layer_frame[["reliable_negative_block", "reliable_negative_sa", "reliable_negative_ff"]].any(axis=None))
    repeated_repair = bool(layer_frame.repeated_repair.any())
    negative_cells = layer_frame[layer_frame[["reliable_negative_block", "reliable_negative_sa", "reliable_negative_ff"]].any(axis=1)]
    result = {
        "cycle": "B", "experiment": "B4", "revision": int(config["revision"]), "source_commit": current_commit(),
        "conditions": len(conditions), "runs": len(registry), "seeds_per_condition": len(config["training"]["seeds"]),
        "causal_rows": len(all_records), "quality_rows": len(all_quality), "matrix_rows": len(all_matrices),
        "parameters": parameters, "native_capture_max_logit_error": max(row["native_capture_max_logit_error"] for row in registry),
        "all_main_conditions_competent": competent, "effective_block_fraction_by_depth": {str(int(k)): float(v) for k, v in fractions.items()},
        "depth_dependent_slack": depth_slack, "reliable_near_zero_utility": repeated_zero,
        "reliable_negative_utility": repeated_negative, "repeated_repair": repeated_repair,
        "reliable_negative_cells": [{"condition": row.condition, "task_family": row.task_family, "layer": int(row.layer), "components": [component for component in ("block", "sa", "ff") if getattr(row, f"reliable_negative_{component}")]} for row in negative_cells.itertuples()],
        "b6_gate_open_from_b4": bool(depth_slack or repeated_zero or repeated_negative or repeated_repair),
        "elapsed_seconds": float(previous_summary.get("elapsed_seconds", time.perf_counter() - started)) if args.postprocess_only else time.perf_counter() - started,
        "postprocess_elapsed_seconds": time.perf_counter() - started if args.postprocess_only else None,
        "interpretation": "Geometry and redundancy remain descriptive; the B6 gate uses replicated causal utility, repair, or competence-preserving depth slack only.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# B4 failure and null-result notes\n\n- Sum-mod-10 is excluded because Cycle A did not establish competence.\n- Effective depth is a single-layer ablation count, not a jointly minimal subnetwork.\n- Repair requires both vector and task-proxy recovery.\n- Pairwise matrices describe redundancy geometry and are not called interference.\n",
        encoding="utf-8",
    )
    metadata = RunMetadata.collect(run_id="cycle-b-b4", model="tiny_residual_decoder", model_variant="baseline_depth_sweep", dataset="synthetic_counterfactual_v1_max_min", seed=0, context_length=int(config["data"]["max_length"]), batch_size=int(config["training"]["batch_size"]), dtype="float32", device=registry[0]["device"])
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(f"B4 complete runs={len(registry)} elapsed_seconds={result['elapsed_seconds']:.3f}\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
