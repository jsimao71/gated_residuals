"""Run Cycle B2 self-attention/feed-forward decomposition and interventions."""

from __future__ import annotations

import argparse
import hashlib
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

from gated_residuals.artifacts import (
    RunMetadata,
    save_manifest,
    save_records_csv,
    save_records_parquet,
)
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import (
    confidence,
    current_commit,
    load_seed,
    make_loader,
    resolve_device,
)
from gated_residuals.residual_dynamics import pairwise_layer_matrices, sa_ff_geometry
from gated_residuals.standard_metrics import target_log_probability


COMPONENT_METRICS = (
    "attention_update_norm",
    "attention_relative_update_norm",
    "attention_state_update_cosine",
    "attention_novelty",
    "attention_dominance",
    "ff_update_norm",
    "ff_relative_update_norm",
    "ff_state_update_cosine",
    "ff_novelty",
    "ff_dominance",
    "attention_ff_cosine",
    "attention_ff_cancellation",
    "combined_update_norm",
    "combined_vs_sum_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b2_sa_ff.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def _safe_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite B2 metric")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cycle_a_reference(seed: int) -> dict[tuple[str, int], dict]:
    path = Path("results/e1/runs") / f"seed-{seed}" / "block_records.csv"
    frame = pd.read_csv(path)
    return {
        (str(row.example_id), int(row.layer)): {
            "full_probability": float(row.full_probability),
            "skipped_probability": float(row.skipped_probability),
        }
        for row in frame.itertuples()
    }


@torch.inference_mode()
def capture_checkpoint(checkpoint: Path, config: dict, output: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]
    model, splits, _, datasets = load_seed(model_config, checkpoint)
    seed = int(payload["seed"])
    variant = str(payload["variant"])
    if variant != "baseline":
        raise RuntimeError(f"B2 confirmatory source must be baseline, got {variant}")
    run_id = f"cycleA-e1-baseline-seed-{seed}"
    device_name = resolve_device(model_config["training"].get("device", "auto"))
    device = torch.device(device_name)
    model.to(device).eval()
    loader = make_loader(datasets["test"], model_config, shuffle=False, seed=seed)
    token_bins = int(config["capture"]["token_bins"])
    random_seed = int(config["capture"]["random_replacement_seed"]) + seed
    reference = _cycle_a_reference(seed)
    component_records: list[dict] = []
    causal_records: list[dict] = []
    final_components = defaultdict(
        lambda: {
            "pre": [[] for _ in range(model.num_layers)],
            "sa": [[] for _ in range(model.num_layers)],
            "after_sa": [[] for _ in range(model.num_layers)],
            "ff": [[] for _ in range(model.num_layers)],
        }
    )
    max_native_parity_error = 0.0
    max_cycle_a_probability_error = 0.0
    max_cycle_a_block_skip_error = 0.0
    max_random_norm_error = 0.0
    started = time.perf_counter()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        untouched = model(ids, mask)
        native = model(ids, mask, capture=True)
        parity_error = float((untouched.logits.float() - native.logits.float()).abs().max())
        max_native_parity_error = max(max_native_parity_error, parity_error)
        if not torch.equal(untouched.logits, native.logits):
            raise RuntimeError(f"native capture parity failed with max error {parity_error}")

        full_logprob = target_log_probability(native.logits, targets)
        full_probability = full_logprob.exp()
        correct = native.logits.argmax(dim=-1) == targets
        interventions = []
        for layer in range(model.num_layers):
            skip_sa = model(ids, mask, skip_attention_layers={layer})
            skip_ff = model(ids, mask, skip_ff_layers={layer})
            skip_block = model(ids, mask, skip_layers={layer})
            random_sa = model(
                ids,
                mask,
                random_attention_layers={layer},
                intervention_seed=random_seed,
                capture=True,
            )
            random_ff = model(
                ids,
                mask,
                random_ff_layers={layer},
                intervention_seed=random_seed,
                capture=True,
            )
            sa_norm_error = float(
                (
                    torch.linalg.vector_norm(
                        random_sa.attention_effective_updates[layer].float(), dim=-1
                    )
                    - torch.linalg.vector_norm(
                        native.attention_candidates[layer].float(), dim=-1
                    )
                )
                .abs()
                .max()
            )
            ff_norm_error = float(
                (
                    torch.linalg.vector_norm(
                        random_ff.ff_effective_updates[layer].float(), dim=-1
                    )
                    - torch.linalg.vector_norm(random_ff.ff_candidates[layer].float(), dim=-1)
                )
                .abs()
                .max()
            )
            max_random_norm_error = max(max_random_norm_error, sa_norm_error, ff_norm_error)
            interventions.append(
                {
                    "skip_sa": target_log_probability(skip_sa.logits, targets),
                    "skip_ff": target_log_probability(skip_ff.logits, targets),
                    "skip_block": target_log_probability(skip_block.logits, targets),
                    "random_sa": target_log_probability(random_sa.logits, targets),
                    "random_ff": target_log_probability(random_ff.logits, targets),
                    "skip_sa_correct": skip_sa.logits.argmax(dim=-1) == targets,
                    "skip_ff_correct": skip_ff.logits.argmax(dim=-1) == targets,
                    "skip_block_correct": skip_block.logits.argmax(dim=-1) == targets,
                    "random_sa_correct": random_sa.logits.argmax(dim=-1) == targets,
                    "random_ff_correct": random_ff.logits.argmax(dim=-1) == targets,
                }
            )

        component_geometry = []
        for layer in range(model.num_layers):
            geometry = sa_ff_geometry(
                native.states[layer],
                native.attention_candidates[layer],
                native.ff_candidates[layer],
            )
            component_geometry.append({name: value.cpu() for name, value in geometry.items()})
        states_cpu = [value.cpu() for value in native.states]
        sa_cpu = [value.cpu() for value in native.attention_candidates]
        after_sa_cpu = [value.cpu() for value in native.states_after_attention]
        ff_cpu = [value.cpu() for value in native.ff_candidates]
        full_logprob_cpu = full_logprob.cpu()
        full_probability_cpu = full_probability.cpu()
        correct_cpu = correct.cpu()
        intervention_cpu = [
            {name: value.cpu() for name, value in values.items()} for values in interventions
        ]

        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = splits["test"][example_index]
            length = int(mask[row].sum())
            for layer in range(model.num_layers):
                values = intervention_cpu[layer]
                ref = reference[(example.example_id, layer)]
                probability_error = abs(_safe_float(full_probability_cpu[row]) - ref["full_probability"])
                skip_probability = _safe_float(values["skip_block"][row].exp())
                skip_error = abs(skip_probability - ref["skipped_probability"])
                max_cycle_a_probability_error = max(max_cycle_a_probability_error, probability_error)
                max_cycle_a_block_skip_error = max(max_cycle_a_block_skip_error, skip_error)
                utility_sa = full_logprob_cpu[row] - values["skip_sa"][row]
                utility_ff = full_logprob_cpu[row] - values["skip_ff"][row]
                utility_block = full_logprob_cpu[row] - values["skip_block"][row]
                utility_random_sa = full_logprob_cpu[row] - values["random_sa"][row]
                utility_random_ff = full_logprob_cpu[row] - values["random_ff"][row]
                final_geometry = component_geometry[layer]
                causal_records.append(
                    {
                        "cycle": "B",
                        "experiment": "B2",
                        "revision": int(config["revision"]),
                        "source_cycle": "A",
                        "source_experiment": "E1",
                        "run_id": run_id,
                        "model_family": "tiny_custom_transformer",
                        "model_variant": variant,
                        "depth": model.num_layers,
                        "width": model.width,
                        "task_family": example.intent,
                        "example_id": example.example_id,
                        "layer": layer,
                        "seed": seed,
                        "full_correct": bool(correct_cpu[row]),
                        "full_target_logprob": _safe_float(full_logprob_cpu[row]),
                        "utility_sa": _safe_float(utility_sa),
                        "utility_ff": _safe_float(utility_ff),
                        "utility_block": _safe_float(utility_block),
                        "utility_random_sa": _safe_float(utility_random_sa),
                        "utility_random_ff": _safe_float(utility_random_ff),
                        "skip_sa_correct": bool(values["skip_sa_correct"][row]),
                        "skip_ff_correct": bool(values["skip_ff_correct"][row]),
                        "skip_block_correct": bool(values["skip_block_correct"][row]),
                        "random_sa_correct": bool(values["random_sa_correct"][row]),
                        "random_ff_correct": bool(values["random_ff_correct"][row]),
                        "attention_ff_cosine": _safe_float(
                            final_geometry["attention_ff_cosine"][row, length - 1]
                        ),
                        "attention_ff_cancellation": _safe_float(
                            final_geometry["attention_ff_cancellation"][row, length - 1]
                        ),
                    }
                )
                for token_index in range(length):
                    token_bin = min(
                        token_bins - 1,
                        int(token_index * token_bins / max(length, 1)),
                    )
                    record = {
                        "cycle": "B",
                        "experiment": "B2",
                        "revision": int(config["revision"]),
                        "run_id": run_id,
                        "model_family": "tiny_custom_transformer",
                        "model_variant": variant,
                        "task_family": example.intent,
                        "example_id": example.example_id,
                        "token_index": token_index,
                        "token_bin": token_bin,
                        "is_final_token": token_index == length - 1,
                        "layer": layer,
                        "seed": seed,
                        "full_correct": bool(correct_cpu[row]),
                        "state_location_sa": "block_residual_pre",
                        "state_location_ff": "residual_after_attention",
                    }
                    for metric in COMPONENT_METRICS:
                        record[metric] = _safe_float(
                            component_geometry[layer][metric][row, token_index]
                        )
                    component_records.append(record)
                final_components[example.intent]["pre"][layer].append(
                    states_cpu[layer][row, length - 1]
                )
                final_components[example.intent]["sa"][layer].append(
                    sa_cpu[layer][row, length - 1]
                )
                final_components[example.intent]["after_sa"][layer].append(
                    after_sa_cpu[layer][row, length - 1]
                )
                final_components[example.intent]["ff"][layer].append(
                    ff_cpu[layer][row, length - 1]
                )

    if max_cycle_a_probability_error > 1e-6 or max_cycle_a_block_skip_error > 1e-6:
        raise RuntimeError(
            "B2 refactor drifted from Cycle A probabilities: "
            f"native={max_cycle_a_probability_error}, block_skip={max_cycle_a_block_skip_error}"
        )
    frame = pd.DataFrame(component_records)
    final_records = frame[frame["is_final_token"]].to_dict("records")
    save_records_parquet(output / "derived" / f"{run_id}-components.parquet", final_records)
    summaries = []
    groups = frame.groupby(
        ["task_family", "layer", "token_bin", "full_correct"], dropna=False
    )
    for keys, group in groups:
        task, layer, token_bin, is_correct = keys
        for metric in COMPONENT_METRICS:
            values = group[metric]
            summaries.append(
                {
                    "run_id": run_id,
                    "model_variant": variant,
                    "seed": seed,
                    "task_family": task,
                    "layer": int(layer),
                    "token_bin": int(token_bin),
                    "correct_partition": bool(is_correct),
                    "metric": metric,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    matrices = []
    for task in sorted(final_components):
        components = final_components[task]
        for prefix, state_key, update_key in (
            ("sa", "pre", "sa"),
            ("ff", "after_sa", "ff"),
        ):
            states = torch.stack([torch.stack(values) for values in components[state_key]])
            updates = torch.stack([torch.stack(values) for values in components[update_key]])
            values = pairwise_layer_matrices(states, updates)
            for matrix_name, matrix in values.items():
                for layer_i in range(model.num_layers):
                    for layer_j in range(model.num_layers):
                        matrices.append(
                            {
                                "run_id": run_id,
                                "model_variant": variant,
                                "seed": seed,
                                "task_family": task,
                                "component": prefix,
                                "matrix": matrix_name,
                                "layer_i": layer_i,
                                "layer_j": layer_j,
                                "value": _safe_float(matrix[layer_i, layer_j]),
                            }
                        )
    registry = {
        "cycle": "B",
        "experiment": "B2",
        "revision": int(config["revision"]),
        "run_id": run_id,
        "model_family": "tiny_custom_transformer",
        "model_variant": variant,
        "depth": model.num_layers,
        "width": model.width,
        "task_family": "synthetic_counterfactual_v1",
        "seed": seed,
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_commit": current_commit(),
        "test_examples": len(splits["test"]),
        "component_observations_computed": len(component_records),
        "causal_observations": len(causal_records),
        "native_capture_max_logit_error": max_native_parity_error,
        "cycle_a_native_max_probability_error": max_cycle_a_probability_error,
        "cycle_a_block_skip_max_probability_error": max_cycle_a_block_skip_error,
        "random_replacement_max_norm_error": max_random_norm_error,
        "elapsed_seconds": time.perf_counter() - started,
        "device": device_name,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return registry, causal_records, summaries, matrices


def summarize_causal(causal_records: list[dict], config: dict) -> tuple[list[dict], dict]:
    frame = pd.DataFrame(causal_records)
    utility_metrics = (
        "utility_sa",
        "utility_ff",
        "utility_block",
        "utility_random_sa",
        "utility_random_ff",
    )
    seed_rows = []
    for keys, group in frame.groupby(["task_family", "layer", "seed"]):
        task, layer, seed = keys
        row = {"task_family": task, "layer": int(layer), "seed": int(seed), "n": len(group)}
        for metric in utility_metrics:
            row[metric] = float(group[metric].mean())
        row["utility_sa_minus_ff"] = row["utility_sa"] - row["utility_ff"]
        row["attention_ff_cosine"] = float(group["attention_ff_cosine"].mean())
        row["attention_ff_cancellation"] = float(group["attention_ff_cancellation"].mean())
        seed_rows.append(row)

    threshold = float(config["statistics"]["negligible_utility"])
    specialization_effect = float(config["statistics"]["specialization_effect"])
    summary_rows = []
    seed_frame = pd.DataFrame(seed_rows)
    for keys, group in seed_frame.groupby(["task_family", "layer"]):
        task, layer = keys
        row = {"task_family": task, "layer": int(layer), "seed_n": len(group)}
        for metric in (*utility_metrics, "utility_sa_minus_ff", "attention_ff_cosine", "attention_ff_cancellation"):
            values = group[metric].tolist()
            mean, low, high = confidence(values, seed=int(layer) + len(metric))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_median"] = float(pd.Series(values).median())
            row[f"{metric}_std"] = float(pd.Series(values).std(ddof=1))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        differences = group["utility_sa_minus_ff"]
        positive = int((differences > specialization_effect).sum())
        negative = int((differences < -specialization_effect).sum())
        if row["utility_sa_minus_ff_ci95_low"] > specialization_effect and positive >= 4:
            row["specialization"] = "SA_more_useful"
        elif row["utility_sa_minus_ff_ci95_high"] < -specialization_effect and negative >= 4:
            row["specialization"] = "FF_more_useful"
        else:
            row["specialization"] = "not_replicated"
        harmful_sa = int((group["utility_sa"] < -threshold).sum()) >= 4
        useful_ff = int((group["utility_ff"] > threshold).sum()) >= 4
        row["intra_block_repair_candidate"] = bool(
            harmful_sa
            and useful_ff
            and row["utility_sa_ci95_high"] < -threshold
            and row["utility_ff_ci95_low"] > threshold
            and row["attention_ff_cosine_mean"] < 0
        )
        summary_rows.append(row)

    learned = seed_frame[seed_frame["task_family"].isin(["maximum", "minimum"])]
    task_differences = []
    for layer in sorted(learned["layer"].unique()):
        layer_frame = learned[learned["layer"] == layer]
        for component in ("utility_sa", "utility_ff"):
            pivot = layer_frame.pivot(index="seed", columns="task_family", values=component)
            values = (pivot["maximum"] - pivot["minimum"]).tolist()
            mean, low, high = confidence(values, seed=500 + int(layer))
            task_differences.append(
                {
                    "layer": int(layer),
                    "component": component.removeprefix("utility_"),
                    "seed_n": len(values),
                    "mean_maximum_minus_minimum": mean,
                    "median_maximum_minus_minimum": float(pd.Series(values).median()),
                    "std_maximum_minus_minimum": float(pd.Series(values).std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "replicated": bool(
                        (low > specialization_effect and sum(v > specialization_effect for v in values) >= 4)
                        or (high < -specialization_effect and sum(v < -specialization_effect for v in values) >= 4)
                    ),
                }
            )
    decision = {
        "specialized_task_layer_cells": sum(
            row["specialization"] != "not_replicated" for row in summary_rows
        ),
        "intra_block_repair_candidate_cells": sum(
            row["intra_block_repair_candidate"] for row in summary_rows
        ),
        "competence_matched_task_conditioned_cells": sum(
            row["replicated"] for row in task_differences
        ),
        "task_differences": task_differences,
    }
    return summary_rows, decision


def plot_b2(causal_records: list[dict], component_summaries: list[dict], output: Path) -> None:
    causal = pd.DataFrame(causal_records)
    utilities = causal.groupby(["task_family", "layer"])[["utility_sa", "utility_ff"]].mean()
    components = pd.DataFrame(component_summaries)
    components = components[
        (components["correct_partition"])
        & (components["metric"].isin(["attention_relative_update_norm", "ff_relative_update_norm"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)
    for task in ("maximum", "minimum", "sum_mod_10"):
        values = utilities.loc[task]
        axes[0].plot(values.index, values["utility_sa"], marker="o", label=f"SA {task}")
        axes[0].plot(values.index, values["utility_ff"], marker="x", linestyle="--", label=f"FF {task}")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Causal target-log-probability utility", xlabel="layer", ylabel="utility")
    axes[0].legend(fontsize=6, ncol=2)
    pivot = components.pivot_table(index="layer", columns="metric", values="mean")
    axes[1].plot(pivot.index, pivot["attention_relative_update_norm"], marker="o", label="SA")
    axes[1].plot(pivot.index, pivot["ff_relative_update_norm"], marker="o", label="FF")
    axes[1].set(title="Relative candidate-write magnitude", xlabel="layer", ylabel="relative norm")
    axes[1].legend()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "b2_sa_ff.png", dpi=180)
    fig.savefig(figure_dir / "b2_sa_ff.pdf")
    plt.close(fig)


def scientific_summary(
    causal_records: list[dict],
    component_summaries: list[dict],
    matrices: list[dict],
    causal_summary: list[dict],
) -> dict:
    causal = pd.DataFrame(causal_records)
    intervention_accuracy = {
        column.removesuffix("_correct"): {
            "mean_across_layer_interventions": float(causal[column].mean()),
            "by_layer": {
                str(int(layer)): float(group[column].mean())
                for layer, group in causal.groupby("layer")
            },
        }
        for column in (
            "full_correct",
            "skip_sa_correct",
            "skip_ff_correct",
            "skip_block_correct",
            "random_sa_correct",
            "random_ff_correct",
        )
    }
    task_utility = (
        causal.groupby("task_family")
        [
            [
                "utility_sa",
                "utility_ff",
                "utility_block",
                "utility_random_sa",
                "utility_random_ff",
                "attention_ff_cosine",
                "attention_ff_cancellation",
            ]
        ]
        .mean()
        .to_dict("index")
    )
    matrix_frame = pd.DataFrame(matrices)
    matrix_summary = {}
    for keys, group in matrix_frame[
        matrix_frame["layer_i"] != matrix_frame["layer_j"]
    ].groupby(["component", "matrix"]):
        component, matrix = keys
        per_seed = group.groupby("seed")["value"].mean().tolist()
        mean, low, high = confidence(per_seed, seed=len(component) + len(matrix))
        matrix_summary[f"{component}_{matrix}"] = {
            "seed_n": len(per_seed),
            "mean": mean,
            "median": float(pd.Series(per_seed).median()),
            "std": float(pd.Series(per_seed).std(ddof=1)),
            "ci95_low": low,
            "ci95_high": high,
        }
    component_frame = pd.DataFrame(component_summaries)
    final_correct = component_frame[
        (component_frame["correct_partition"])
        & (component_frame["token_bin"] == component_frame["token_bin"].max())
    ]
    geometry_summary = {}
    for metric in (
        "attention_relative_update_norm",
        "ff_relative_update_norm",
        "attention_state_update_cosine",
        "ff_state_update_cosine",
        "attention_ff_cosine",
        "attention_ff_cancellation",
    ):
        geometry_summary[metric] = {
            str(int(layer)): float(group["mean"].mean())
            for layer, group in final_correct[final_correct["metric"] == metric].groupby("layer")
        }
    specialized = [
        {
            "task_family": row["task_family"],
            "layer": int(row["layer"]),
            "specialization": row["specialization"],
            "sa_minus_ff_mean": row["utility_sa_minus_ff_mean"],
            "ci95_low": row["utility_sa_minus_ff_ci95_low"],
            "ci95_high": row["utility_sa_minus_ff_ci95_high"],
        }
        for row in causal_summary
        if row["specialization"] != "not_replicated"
    ]
    return {
        "intervention_accuracy": intervention_accuracy,
        "task_mean_utility": task_utility,
        "offdiagonal_component_matrices": matrix_summary,
        "correct_final_token_geometry": geometry_summary,
        "replicated_specialization_cells": specialized,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = read_yaml(config_path)
    output = Path(args.output or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "config_snapshot.yaml")
    metadata = RunMetadata.collect(
        run_id="cycle-b-b2-sa-ff",
        model="tiny_residual_decoder",
        model_variant="baseline",
        dataset="synthetic_counterfactual_v1",
        seed=0,
        intervention="native_skip_sa_skip_ff_skip_block_norm_matched_random",
        context_length=56,
        batch_size=64,
        hook_locations={
            "residual_pre": "TinyOutput.states[layer]",
            "attention_candidate_update": "TinyOutput.attention_candidates[layer]",
            "residual_after_attention": "TinyOutput.states_after_attention[layer]",
            "ff_candidate_update": "TinyOutput.ff_candidates[layer]",
            "residual_post": "TinyOutput.states[layer + 1]",
        },
        dtype="float32",
        device=resolve_device("auto"),
    )
    save_manifest(output / "manifest.json", metadata, config)
    checkpoints = sorted(Path().glob(config["checkpoint_glob"]))
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise RuntimeError("no B2 checkpoints found")
    registry_rows, causal_records, component_summaries, matrices = [], [], [], []
    log_lines = [f"B2 start checkpoints={len(checkpoints)} source_commit={current_commit()}"]
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
        registry, causal, summaries, matrix_rows = capture_checkpoint(checkpoint, config, output)
        registry_rows.append(registry)
        causal_records.extend(causal)
        component_summaries.extend(summaries)
        matrices.extend(matrix_rows)
        log_lines.append(
            f"complete {registry['run_id']} seconds={registry['elapsed_seconds']:.3f} "
            f"native_error={registry['native_capture_max_logit_error']:.3g}"
        )
    causal_summary, decision = summarize_causal(causal_records, config)
    save_records_csv(output / "checkpoint_registry.csv", registry_rows)
    save_records_parquet(output / "causal_interventions.parquet", causal_records)
    save_records_csv(output / "causal_interventions.csv", causal_records)
    save_records_csv(output / "component_layer_token_summary.csv", component_summaries)
    save_records_csv(output / "pairwise_component_matrices.csv", matrices)
    save_records_csv(output / "causal_summary.csv", causal_summary)
    save_records_csv(output / "task_conditioned_summary.csv", decision["task_differences"])
    plot_b2(causal_records, component_summaries, output)
    summary = {
        "cycle": "B",
        "experiment": "B2",
        "revision": int(config["revision"]),
        "source_commit": current_commit(),
        "checkpoints": len(checkpoints),
        "component_observations_computed": sum(
            row["component_observations_computed"] for row in registry_rows
        ),
        "causal_observations": len(causal_records),
        "native_capture_max_logit_error": max(
            row["native_capture_max_logit_error"] for row in registry_rows
        ),
        "cycle_a_native_max_probability_error": max(
            row["cycle_a_native_max_probability_error"] for row in registry_rows
        ),
        "cycle_a_block_skip_max_probability_error": max(
            row["cycle_a_block_skip_max_probability_error"] for row in registry_rows
        ),
        "random_replacement_max_norm_error": max(
            row["random_replacement_max_norm_error"] for row in registry_rows
        ),
        "zero_replacement_aliases_skip": True,
        **{key: value for key, value in decision.items() if key != "task_differences"},
        "scientific_summary": scientific_summary(
            causal_records, component_summaries, matrices, causal_summary
        ),
        "interpretation": (
            "SA/FF geometry is descriptive; sublayer utility uses explicit intervention. "
            "Skip-SA recomputes FF from the intervened post-SA state."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# B2 failure and null-result notes\n\n"
        "- Zero replacement is mathematically identical to the registered skip-SA/skip-FF path; "
        "the alias is tested rather than duplicated as another forward.\n"
        "- Skip-SA recomputes the FF candidate from the intervened residual-after-attention state.\n"
        "- Norm-matched random replacements are exploratory single deterministic draws.\n"
        "- Negative SA--FF cosine or cancellation is not called repair without causal utility.\n",
        encoding="utf-8",
    )
    log_lines.append(f"B2 complete causal_observations={len(causal_records)}")
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
