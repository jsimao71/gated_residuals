"""Run Cycle B1's residual-dynamics atlas on frozen Cycle A checkpoints."""

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
from gated_residuals.attention_dilution import attention_metrics, evidence_mass
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import (
    confidence,
    current_commit,
    load_seed,
    make_loader,
    resolve_device,
)
from gated_residuals.residual_dynamics import (
    cancellation_score,
    pairwise_layer_matrices,
    update_geometry,
)
from gated_residuals.standard_metrics import logit_margin, target_log_probability
from gated_residuals.synthetic import NUMBER_WORDS, tokenize


METRIC_INVENTORY = [
    ("state magnitude", "residual_norm", "implemented_and_run"),
    ("candidate update magnitude", "candidate_update_norm", "implemented_and_run"),
    ("effective update magnitude", "effective_update_norm", "implemented_and_run"),
    ("relative update magnitude", "relative_update_norm", "implemented_and_run"),
    ("state/update alignment", "candidate_cosine", "implemented_and_run"),
    ("inter-update alignment", "adjacent_update_cosine", "implemented_and_run"),
    ("cancellation/reinforcement", "adjacent_cancellation", "implemented_and_run"),
    ("minimum novelty", "candidate_novelty", "implemented_and_run"),
    ("recent-update-subspace novelty", "recent_subspace_novelty", "deferred_to_B3"),
    ("dominance", "candidate_dominance", "implemented_and_run"),
    ("effective displacement", "effective_displacement", "implemented_and_run"),
    ("representation direction drift", "representation_direction_cosine", "implemented_and_run"),
    ("residual-state cosine matrix", "residual_state_cosine", "implemented_and_run"),
    ("update cosine matrix", "update_cosine", "implemented_and_run"),
    ("CKA matrices", "residual_state_cka/update_cka", "implemented_and_run"),
    ("RSA matrices", "residual_state_rsa/update_rsa", "implemented_and_run"),
    ("attention concentration", "entropy/top-k/support/sink", "implemented_and_run"),
    ("synthetic evidence mass", "attention_evidence_mass", "implemented_and_run"),
    ("per-head output norm", "head_output_norm", "requires_B2_capture"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b1_residual_atlas.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_experiment(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for experiment in ("e1", "e3", "e4", "e5"):
        if experiment in parts:
            return experiment.upper()
    raise ValueError(f"cannot identify source experiment for {path}")


def discover_checkpoints(config: dict) -> list[Path]:
    found: set[Path] = set()
    for pattern in config["checkpoint_globs"]:
        found.update(Path().glob(pattern))
    checkpoints = sorted(path for path in found if path.is_file())
    if not checkpoints:
        raise RuntimeError("no Cycle A checkpoints matched the B1 configuration")
    return checkpoints


def _evidence_mask(example, max_length: int, device: torch.device) -> torch.Tensor:
    tokens = tokenize(example.prompt)[-max_length:]
    needle = [NUMBER_WORDS[value] for value in example.content]
    mask = torch.zeros(len(tokens), dtype=torch.bool, device=device)
    for start in range(len(tokens) - len(needle) + 1):
        if tokens[start : start + len(needle)] == needle:
            mask[start : start + len(needle)] = True
            return mask
    raise RuntimeError(f"content tokens were not found in {example.example_id}")


def _head_similarity(attention: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(attention.float(), dim=-1)
    similarity = normalized @ normalized.T
    heads = similarity.shape[0]
    if heads == 1:
        return torch.ones(1, device=attention.device)
    return (similarity.sum(dim=-1) - 1.0) / (heads - 1)


def _safe_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite derived B1 metric")
    return result


@torch.inference_mode()
def capture_checkpoint(
    checkpoint: Path,
    stage_config: dict,
    output: Path,
) -> tuple[dict, list[dict], list[dict]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]
    model, splits, vocabulary, datasets = load_seed(model_config, checkpoint)
    seed = int(payload["seed"])
    variant = str(payload["variant"])
    experiment = _source_experiment(checkpoint)
    run_id = f"cycleA-{experiment.lower()}-{variant}-seed-{seed}"
    device_name = resolve_device(model_config["training"].get("device", "auto"))
    device = torch.device(device_name)
    model.to(device).eval()
    loader = make_loader(datasets["test"], model_config, shuffle=False, seed=seed)
    token_bins = int(stage_config["capture"]["token_bins"])
    topk = int(stage_config["capture"]["attention_topk"])
    residual_records: list[dict] = []
    attention_records: list[dict] = []
    final_states: dict[str, list[list[torch.Tensor]]] = defaultdict(
        lambda: [[] for _ in range(model.num_layers)]
    )
    final_updates: dict[str, list[list[torch.Tensor]]] = defaultdict(
        lambda: [[] for _ in range(model.num_layers)]
    )
    started = time.perf_counter()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        native = model(ids, mask, capture=True)
        target_logprob = target_log_probability(native.logits, targets)
        losses = F.cross_entropy(native.logits, targets, reduction="none")
        margins = logit_margin(native.logits, targets)
        correct = native.logits.argmax(dim=-1) == targets
        skip_logprobs = []
        for layer in range(model.num_layers):
            skipped = model(ids, mask, skip_layers={layer})
            skip_logprobs.append(target_log_probability(skipped.logits, targets))

        batch_geometry = []
        for layer in range(model.num_layers):
            candidate = update_geometry(native.states[layer], native.candidates[layer])
            effective = update_geometry(native.states[layer], native.effective_updates[layer])
            batch_geometry.append((candidate, effective))

        # Transfer each captured/derived tensor once. Calling ``float(cuda_scalar)`` in
        # the nested row/layer/token/head loops would otherwise synchronize thousands
        # of times and dominate this exhaustive analysis.
        states_cpu = [value.cpu() for value in native.states]
        candidates_cpu = [value.cpu() for value in native.candidates]
        gates_cpu = [value.cpu() for value in native.gates]
        attention_cpu = [value.cpu() for value in native.attention]
        geometry_cpu = [
            (
                {name: value.cpu() for name, value in candidate.items()},
                {name: value.cpu() for name, value in effective.items()},
            )
            for candidate, effective in batch_geometry
        ]
        target_logprob_cpu = target_logprob.cpu()
        losses_cpu = losses.cpu()
        margins_cpu = margins.cpu()
        correct_cpu = correct.cpu()
        skip_logprobs_cpu = [value.cpu() for value in skip_logprobs]

        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = splits["test"][example_index]
            length = int(mask[row].sum())
            evidence = _evidence_mask(
                example, int(model_config["data"]["max_length"]), torch.device("cpu")
            )
            if evidence.numel() != length:
                raise RuntimeError("evidence mask and non-padding token length differ")
            for layer in range(model.num_layers):
                candidate_geometry, effective_geometry = geometry_cpu[layer]
                gate = gates_cpu[layer][row].flatten()[0]
                utility = target_logprob_cpu[row] - skip_logprobs_cpu[layer][row]
                attention = attention_cpu[layer][row, :, :length, :length]
                per_head_attention = attention_metrics(attention, topk=topk)
                per_head_evidence = evidence_mass(attention, evidence)
                for token_index in range(length):
                    token_bin = min(
                        token_bins - 1,
                        int(token_index * token_bins / max(length, 1)),
                    )
                    if layer == 0:
                        adjacent_cosine = float("nan")
                        adjacent_cancel = float("nan")
                    else:
                        previous = candidates_cpu[layer - 1][row, token_index]
                        current = candidates_cpu[layer][row, token_index]
                        adjacent_cosine = _safe_float(
                            F.cosine_similarity(previous.float(), current.float(), dim=0)
                        )
                        adjacent_cancel = _safe_float(cancellation_score(previous, current))
                    residual_records.append(
                        {
                            "cycle": "B",
                            "experiment": "B1",
                            "revision": int(stage_config["revision"]),
                            "source_cycle": "A",
                            "source_experiment": experiment,
                            "run_id": run_id,
                            "model": "tiny_residual_decoder",
                            "model_family": "tiny_custom_transformer",
                            "model_variant": variant,
                            "depth": model.num_layers,
                            "width": model.width,
                            "dataset": "synthetic_counterfactual_v1",
                            "task_family": example.intent,
                            "example_id": example.example_id,
                            "token_index": token_index,
                            "is_final_token": token_index == length - 1,
                            "token_bin": token_bin,
                            "layer": layer,
                            "head": -1,
                            "state_location": "block_residual_pre",
                            "intervention": "native",
                            "seed": seed,
                            "checkpoint": str(checkpoint).replace("\\", "/"),
                            "quality_metric": "target_log_probability",
                            "loss": _safe_float(losses_cpu[row]),
                            "target_logprob": _safe_float(target_logprob_cpu[row]),
                            "logit_margin": _safe_float(margins_cpu[row]),
                            "full_correct": bool(correct_cpu[row]),
                            "gate": _safe_float(gate),
                            "candidate_update_norm": _safe_float(
                                candidate_geometry["update_norm"][row, token_index]
                            ),
                            "effective_update_norm": _safe_float(
                                effective_geometry["update_norm"][row, token_index]
                            ),
                            "residual_norm": _safe_float(
                                candidate_geometry["residual_norm"][row, token_index]
                            ),
                            "relative_update_norm": _safe_float(
                                candidate_geometry["relative_update_norm"][row, token_index]
                            ),
                            "candidate_cosine": _safe_float(
                                candidate_geometry["state_update_cosine"][row, token_index]
                            ),
                            "effective_cosine": _safe_float(
                                effective_geometry["state_update_cosine"][row, token_index]
                            ),
                            "candidate_novelty": _safe_float(
                                candidate_geometry["novelty"][row, token_index]
                            ),
                            "candidate_dominance": _safe_float(
                                candidate_geometry["dominance"][row, token_index]
                            ),
                            "effective_displacement": _safe_float(
                                effective_geometry["effective_displacement"][row, token_index]
                            ),
                            "representation_direction_cosine": _safe_float(
                                effective_geometry["representation_direction_cosine"][row, token_index]
                            ),
                            "adjacent_update_cosine": adjacent_cosine,
                            "adjacent_cancellation": adjacent_cancel,
                            "attention_entropy": _safe_float(
                                per_head_attention["attention_entropy"][:, token_index].mean()
                            ),
                            "attention_top1_mass": _safe_float(
                                per_head_attention["attention_top1_mass"][:, token_index].mean()
                            ),
                            "attention_topk_mass": _safe_float(
                                per_head_attention["attention_topk_mass"][:, token_index].mean()
                            ),
                            "attention_effective_support": _safe_float(
                                per_head_attention["attention_effective_support"][:, token_index].mean()
                            ),
                            "attention_sink_score": _safe_float(
                                per_head_attention["attention_sink_ratio"][:, token_index].mean()
                            ),
                            "attention_evidence_mass": _safe_float(
                                per_head_evidence[:, token_index].mean()
                            ),
                            "block_utility": _safe_float(utility),
                            "repair_score": float("nan"),
                        }
                    )
                    similarity = _head_similarity(attention[:, token_index])
                    for head in range(attention.shape[0]):
                        attention_records.append(
                            {
                                "cycle": "B",
                                "experiment": "B1",
                                "revision": int(stage_config["revision"]),
                                "source_experiment": experiment,
                                "run_id": run_id,
                                "model_family": "tiny_custom_transformer",
                                "model_variant": variant,
                                "task_family": example.intent,
                                "example_id": example.example_id,
                                "token_index": token_index,
                                "token_bin": token_bin,
                                "layer": layer,
                                "head": head,
                                "seed": seed,
                                "full_correct": bool(correct_cpu[row]),
                                "attention_entropy": _safe_float(
                                    per_head_attention["attention_entropy"][head, token_index]
                                ),
                                "attention_top1_mass": _safe_float(
                                    per_head_attention["attention_top1_mass"][head, token_index]
                                ),
                                "attention_topk_mass": _safe_float(
                                    per_head_attention["attention_topk_mass"][head, token_index]
                                ),
                                "attention_effective_support": _safe_float(
                                    per_head_attention["attention_effective_support"][head, token_index]
                                ),
                                "attention_sink_score": _safe_float(
                                    per_head_attention["attention_sink_ratio"][head, token_index]
                                ),
                                "attention_evidence_mass": _safe_float(
                                    per_head_evidence[head, token_index]
                                ),
                                "within_layer_head_similarity": _safe_float(similarity[head]),
                            }
                        )
                final_states[example.intent][layer].append(
                    states_cpu[layer][row, length - 1]
                )
                final_updates[example.intent][layer].append(
                    candidates_cpu[layer][row, length - 1]
                )

    derived = output / "derived"
    residual_frame = pd.DataFrame(residual_records)
    final_residual_records = residual_frame[residual_frame["is_final_token"]].to_dict("records")
    save_records_parquet(
        derived / f"{run_id}-final-token-residual.parquet", final_residual_records
    )
    attention_frame = pd.DataFrame(attention_records)
    attention_metric_columns = [
        "attention_entropy",
        "attention_top1_mass",
        "attention_topk_mass",
        "attention_effective_support",
        "attention_sink_score",
        "attention_evidence_mass",
        "within_layer_head_similarity",
    ]
    attention_summaries = []
    attention_groups = attention_frame.groupby(
        [
            "source_experiment",
            "model_variant",
            "task_family",
            "layer",
            "head",
            "token_bin",
            "full_correct",
        ],
        dropna=False,
    )
    for keys, frame in attention_groups:
        source, grouped_variant, task, layer, head, token_bin, is_correct = keys
        for metric in attention_metric_columns:
            values = frame[metric]
            attention_summaries.append(
                {
                    "run_id": run_id,
                    "source_experiment": source,
                    "model_variant": grouped_variant,
                    "seed": seed,
                    "task_family": task,
                    "layer": int(layer),
                    "head": int(head),
                    "token_bin": int(token_bin),
                    "correct_partition": bool(is_correct),
                    "metric": metric,
                    "n": int(values.count()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    save_records_parquet(
        derived / f"{run_id}-attention-summary.parquet", attention_summaries
    )
    grouped = residual_frame.groupby(
        ["task_family", "layer", "token_bin", "full_correct"], dropna=False
    )
    metric_columns = [
        "residual_norm",
        "candidate_update_norm",
        "effective_update_norm",
        "relative_update_norm",
        "candidate_cosine",
        "candidate_novelty",
        "candidate_dominance",
        "effective_displacement",
        "representation_direction_cosine",
        "attention_entropy",
        "attention_evidence_mass",
    ]
    run_summaries = []
    for keys, frame in grouped:
        task, layer, token_bin, is_correct = keys
        for metric in metric_columns:
            values = frame[metric]
            run_summaries.append(
                {
                    "run_id": run_id,
                    "source_experiment": experiment,
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
    matrix_records = []
    for task in sorted(final_states):
        states = torch.stack([torch.stack(layer) for layer in final_states[task]])
        updates = torch.stack([torch.stack(layer) for layer in final_updates[task]])
        matrices = pairwise_layer_matrices(states, updates)
        for matrix_name, matrix in matrices.items():
            for layer_i in range(model.num_layers):
                for layer_j in range(model.num_layers):
                    matrix_records.append(
                        {
                            "run_id": run_id,
                            "source_experiment": experiment,
                            "model_variant": variant,
                            "seed": seed,
                            "task_family": task,
                            "token_scope": "final_nonpadding",
                            "matrix": matrix_name,
                            "layer_i": layer_i,
                            "layer_j": layer_j,
                            "value": _safe_float(matrix[layer_i, layer_j]),
                        }
                    )
    accuracy = float(residual_frame[["example_id", "full_correct"]].drop_duplicates()["full_correct"].mean())
    registry = {
        "cycle": "B",
        "experiment": "B1",
        "revision": int(stage_config["revision"]),
        "source_cycle": "A",
        "source_experiment": experiment,
        "run_id": run_id,
        "model_family": "tiny_custom_transformer",
        "model_variant": variant,
        "depth": model.num_layers,
        "width": model.width,
        "task_family": "synthetic_counterfactual_v1",
        "seed": seed,
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "source_commit": current_commit(),
        "test_examples": len(splits["test"]),
        "test_accuracy": accuracy,
        "residual_observations_computed": len(residual_records),
        "residual_rows_persisted": len(final_residual_records),
        "attention_head_observations_computed": len(attention_records),
        "attention_summary_rows_persisted": len(attention_summaries),
        "elapsed_seconds": time.perf_counter() - started,
        "device": device_name,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return registry, run_summaries, matrix_records


def _seed_summary(run_summaries: list[dict]) -> list[dict]:
    frame = pd.DataFrame(run_summaries)
    selected = frame[
        (frame["correct_partition"])
        & (frame["token_bin"] == frame["token_bin"].max())
    ]
    rows = []
    keys = ["model_variant", "task_family", "layer", "metric"]
    for group_keys, group in selected.groupby(keys):
        variant, task, layer, metric = group_keys
        per_seed = group.groupby("seed")["mean"].mean().tolist()
        mean, low, high = confidence(per_seed, seed=int(layer) + len(rows))
        rows.append(
            {
                "model_variant": variant,
                "task_family": task,
                "layer": int(layer),
                "metric": metric,
                "seed_n": len(per_seed),
                "mean": mean,
                "median": float(pd.Series(per_seed).median()),
                "std": float(pd.Series(per_seed).std(ddof=1)) if len(per_seed) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows


def _correctness_contrasts(run_summaries: list[dict]) -> list[dict]:
    """Paired correct-minus-incorrect descriptive contrasts across checkpoint seeds."""
    frame = pd.DataFrame(run_summaries)
    index = [
        "run_id",
        "model_variant",
        "seed",
        "task_family",
        "layer",
        "token_bin",
        "metric",
    ]
    means = frame.pivot_table(index=index, columns="correct_partition", values="mean")
    counts = frame.pivot_table(index=index, columns="correct_partition", values="n", aggfunc="sum")
    if False not in means or True not in means:
        raise RuntimeError("B1 correctness contrast requires both outcome partitions")
    paired = means.dropna(subset=[False, True]).copy()
    paired["delta"] = paired[True] - paired[False]
    paired["n_correct"] = counts.loc[paired.index, True]
    paired["n_incorrect"] = counts.loc[paired.index, False]
    paired = paired.reset_index()
    rows = []
    group_keys = ["model_variant", "task_family", "layer", "token_bin", "metric"]
    for keys, group in paired.groupby(group_keys):
        variant, task, layer, token_bin, metric = keys
        per_seed = group.groupby("seed")["delta"].mean().tolist()
        mean, low, high = confidence(per_seed, seed=int(layer) * 100 + int(token_bin))
        std = float(pd.Series(per_seed).std(ddof=1)) if len(per_seed) > 1 else 0.0
        rows.append(
            {
                "model_variant": variant,
                "task_family": task,
                "layer": int(layer),
                "token_bin": int(token_bin),
                "metric": metric,
                "seed_n": len(per_seed),
                "mean_correct_minus_incorrect": mean,
                "median_correct_minus_incorrect": float(pd.Series(per_seed).median()),
                "std_correct_minus_incorrect": std,
                "ci95_low": low,
                "ci95_high": high,
                "paired_effect_size_dz": mean / std if std > 0 else 0.0,
                "raw_n_correct": int(group["n_correct"].sum()),
                "raw_n_incorrect": int(group["n_incorrect"].sum()),
            }
        )
    return rows


def _plot_atlas(run_summaries: list[dict], output: Path) -> None:
    frame = pd.DataFrame(run_summaries)
    frame = frame[frame["correct_partition"]]
    metrics = [
        "relative_update_norm",
        "candidate_cosine",
        "candidate_novelty",
        "representation_direction_cosine",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), constrained_layout=True)
    for axis, metric in zip(axes, metrics):
        selected = frame[frame["metric"] == metric]
        pivot = selected.pivot_table(index="layer", columns="token_bin", values="mean")
        image = axis.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("relative token-position bin")
        axis.set_ylabel("layer")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns)
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "b1_layer_token_atlas.png", dpi=180)
    fig.savefig(figure_dir / "b1_layer_token_atlas.pdf")
    plt.close(fig)


def _scientific_summary(output: Path, matrix_records: list[dict]) -> dict:
    """Build paper-facing seed/checkpoint summaries from persisted derived records."""
    matrices = pd.DataFrame(matrix_records)
    matrix_summary = {}
    for matrix_name in matrices["matrix"].unique():
        selected = matrices[
            (matrices["matrix"] == matrix_name)
            & (matrices["layer_i"] != matrices["layer_j"])
        ]
        per_run = selected.groupby("run_id")["value"].mean().tolist()
        mean, low, high = confidence(per_run, seed=len(matrix_name))
        matrix_summary[matrix_name] = {
            "checkpoint_n": len(per_run),
            "mean": mean,
            "median": float(pd.Series(per_run).median()),
            "std": float(pd.Series(per_run).std(ddof=1)),
            "ci95_low": low,
            "ci95_high": high,
        }

    final_frames = [
        pd.read_parquet(path)
        for path in sorted((output / "derived").glob("*-final-token-residual.parquet"))
    ]
    final = pd.concat(final_frames, ignore_index=True)
    baseline = final[final["model_variant"] == "baseline"]
    baseline_by_layer = {}
    for metric in (
        "relative_update_norm",
        "candidate_cosine",
        "candidate_novelty",
        "candidate_dominance",
        "representation_direction_cosine",
        "attention_entropy",
        "attention_evidence_mass",
    ):
        baseline_by_layer[metric] = {}
        for layer in sorted(baseline["layer"].unique()):
            per_seed = (
                baseline[baseline["layer"] == layer].groupby("seed")[metric].mean().tolist()
            )
            mean, low, high = confidence(per_seed, seed=int(layer) + len(metric))
            baseline_by_layer[metric][str(int(layer))] = {
                "seed_n": len(per_seed),
                "mean": mean,
                "median": float(pd.Series(per_seed).median()),
                "std": float(pd.Series(per_seed).std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
            }

    effective_ratio_by_variant = {}
    ratio_frame = final[["run_id", "model_variant"]].copy()
    ratio_frame["ratio"] = final["effective_update_norm"] / final[
        "candidate_update_norm"
    ].clip(lower=1e-8)
    for variant, selected in ratio_frame.groupby("model_variant"):
        per_run = selected.groupby("run_id")["ratio"].mean().tolist()
        mean, low, high = confidence(per_run, seed=50 + len(variant))
        effective_ratio_by_variant[variant] = {
            "checkpoint_n": len(per_run),
            "mean": mean,
            "median": float(pd.Series(per_run).median()),
            "std": float(pd.Series(per_run).std(ddof=1)),
            "ci95_low": low,
            "ci95_high": high,
        }
    return {
        "baseline_final_token_by_layer": baseline_by_layer,
        "cross_checkpoint_offdiagonal_layer_matrices": matrix_summary,
        "effective_to_candidate_norm_ratio": effective_ratio_by_variant,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = read_yaml(config_path)
    output = Path(args.output or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "config_snapshot.yaml")
    save_records_csv(
        output / "metric_inventory.csv",
        [
            {"metric_family": family, "implementation": implementation, "status": status}
            for family, implementation, status in METRIC_INVENTORY
        ],
    )
    metadata = RunMetadata.collect(
        run_id="cycle-b-b1-residual-atlas",
        model="tiny_residual_decoder",
        model_variant="all_cycle_a_variants",
        dataset="synthetic_counterfactual_v1",
        seed=0,
        intervention="native_and_single_block_skip",
        context_length=56,
        batch_size=64,
        hook_locations={
            "residual_pre": "TinyOutput.states[layer]",
            "candidate_update": "TinyOutput.candidates[layer]",
            "effective_update": "TinyOutput.effective_updates[layer]",
            "residual_post": "TinyOutput.states[layer + 1]",
            "attention_weights": "TinyOutput.attention[layer]",
        },
        gate_tensor_semantics="[batch, 1] scalar block gate, repeated across tokens",
        dtype="float32",
        device=resolve_device("auto"),
    )
    save_manifest(output / "manifest.json", metadata, config)
    checkpoints = discover_checkpoints(config)
    if args.max_checkpoints is not None:
        if args.max_checkpoints <= 0:
            raise ValueError("--max-checkpoints must be positive")
        checkpoints = checkpoints[: args.max_checkpoints]
    registry_rows: list[dict] = []
    run_summaries: list[dict] = []
    matrix_records: list[dict] = []
    log_lines = [f"B1 start checkpoints={len(checkpoints)} source_commit={current_commit()}"]
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
        registry, summaries, matrices = capture_checkpoint(checkpoint, config, output)
        registry_rows.append(registry)
        run_summaries.extend(summaries)
        matrix_records.extend(matrices)
        log_lines.append(
            f"complete run_id={registry['run_id']} "
            f"residual_rows={registry['residual_observations_computed']} "
            f"attention_rows={registry['attention_head_observations_computed']} "
            f"seconds={registry['elapsed_seconds']:.3f}"
        )
    save_records_csv(output / "checkpoint_registry.csv", registry_rows)
    save_records_csv(output / "layer_token_summary.csv", run_summaries)
    seed_summary = _seed_summary(run_summaries)
    save_records_csv(output / "seed_summary.csv", seed_summary)
    save_records_csv(
        output / "correctness_contrasts.csv", _correctness_contrasts(run_summaries)
    )
    save_records_csv(output / "pairwise_layer_matrices.csv", matrix_records)
    _plot_atlas(run_summaries, output)
    accuracy_by_variant = {}
    for variant, frame in pd.DataFrame(registry_rows).groupby("model_variant"):
        values = frame["test_accuracy"]
        accuracy_by_variant[variant] = {
            "runs": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    summary = {
        "cycle": "B",
        "experiment": "B1",
        "revision": int(config["revision"]),
        "source_cycle": "A",
        "source_commit": current_commit(),
        "checkpoints": len(registry_rows),
        "variants": sorted(accuracy_by_variant),
        "residual_observations_computed": sum(
            row["residual_observations_computed"] for row in registry_rows
        ),
        "residual_rows_persisted": sum(
            row["residual_rows_persisted"] for row in registry_rows
        ),
        "attention_head_observations_computed": sum(
            row["attention_head_observations_computed"] for row in registry_rows
        ),
        "attention_summary_rows_persisted": sum(
            row["attention_summary_rows_persisted"] for row in registry_rows
        ),
        "pairwise_matrix_cells": len(matrix_records),
        "accuracy_by_variant": accuracy_by_variant,
        "scientific_summary": _scientific_summary(output, matrix_records),
        "nonfinite_required_metrics": 0,
        "native_instrumentation_valid": True,
        "raw_activations_persisted": False,
        "interpretation": (
            "B1 is a descriptive residual atlas. Geometry alone is not evidence of interference; "
            "Cycle A causal conclusions remain unchanged."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# B1 failure and null-result notes\n\n"
        "- No raw activation tensor was persisted; only derived per-example/token/layer/head metrics.\n"
        "- Cycle A's tiny block capture combines SA and FF writes, so per-head output norms and "
        "SA/FF causal roles are intentionally deferred to B2.\n"
        "- Minimum novelty is the registered $1-\\cos^2(x,\\Delta)$ statistic; a recent-update "
        "subspace version is deferred to B3.\n"
        "- Cancellation and anti-alignment remain descriptive geometry, not interference claims.\n",
        encoding="utf-8",
    )
    log_lines.append(
        f"B1 complete residual_observations={summary['residual_observations_computed']} "
        f"attention_head_observations={summary['attention_head_observations_computed']}"
    )
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
