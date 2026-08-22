"""Run Cycle B3 token/depth stability and amplification-repair analysis."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gated_residuals.artifacts import RunMetadata, save_manifest, save_records_csv, save_records_parquet
from gated_residuals.attention_dilution import attention_metrics
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, current_commit, load_seed, make_loader, resolve_device
from gated_residuals.residual_dynamics import amplification_repair
from gated_residuals.standard_metrics import target_log_probability
from gated_residuals.temporal_stability import (
    autocorrelation,
    correlation_length,
    covariance_drift,
    eigenspectrum_drift,
    first_second_differences,
    lagged_cross_correlation,
    linear_cka,
    mean_shift,
    principal_subspace_drift,
    rbf_mmd,
    representational_similarity,
    rolling_moments,
    stable_rank,
    wasserstein_1d,
)


SERIES = ("residual_norm", "block_update_norm", "sa_update_norm", "ff_update_norm", "attention_entropy")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b3_stability.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def safe(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite B3 required metric")
    return result


def scalar_features(values: torch.Tensor, prefix: str, config: dict) -> dict:
    values = values.detach().float().flatten()
    differences = first_second_differences(values)
    max_lag = min(int(config["capture"]["max_lag"]), values.numel() - 2)
    acf = autocorrelation(values.view(1, -1, 1), max_lag)[0]
    output = {
        f"{prefix}_mean": safe(values.mean()),
        f"{prefix}_variance": safe(values.var(unbiased=False)),
        f"{prefix}_first_difference_abs_mean": safe(differences["first_difference"].abs().mean()),
        f"{prefix}_second_difference_abs_mean": safe(differences["second_difference"].abs().mean()),
        f"{prefix}_acf_lag1": safe(acf[1]),
        f"{prefix}_correlation_length": safe(correlation_length(acf.unsqueeze(0))[0]),
    }
    split = values.numel() // 2
    first, second = values[:split], values[-split:]
    output[f"{prefix}_window_mean_shift"] = safe((second.mean() - first.mean()).abs())
    output[f"{prefix}_wasserstein"] = safe(wasserstein_1d(first, second))
    for window in config["capture"]["rolling_windows"]:
        window = int(window)
        if window > values.numel():
            continue
        moments = rolling_moments(values.view(1, -1, 1), window)
        output[f"{prefix}_w{window}_rolling_variance"] = safe(moments["variance"].mean())
        output[f"{prefix}_w{window}_rolling_abs_skewness"] = safe(moments["skewness"].abs().mean())
        output[f"{prefix}_w{window}_rolling_abs_kurtosis"] = safe(moments["excess_kurtosis"].abs().mean())
    return output


def vector_window_features(values: torch.Tensor, rank: int) -> dict:
    values = values.detach().float()
    split = values.shape[0] // 2
    size = min(split, values.shape[0] - split)
    first, second = values[:size], values[-size:]
    usable_rank = min(rank, size - 1, values.shape[1])
    return {
        "residual_vector_mean_shift": safe(mean_shift(first, second)),
        "residual_covariance_drift": safe(covariance_drift(first, second)),
        "residual_wasserstein_norm": safe(
            wasserstein_1d(torch.linalg.vector_norm(first, dim=-1), torch.linalg.vector_norm(second, dim=-1))
        ),
        "residual_mmd": safe(rbf_mmd(first, second)),
        "residual_subspace_drift": safe(principal_subspace_drift(first, second, usable_rank)),
        "residual_eigenspectrum_drift": safe(eigenspectrum_drift(first, second)),
        "residual_window_cka": safe(linear_cka(first, second)),
        "residual_window_rsa": safe(representational_similarity(first, second)),
        "residual_first_stable_rank": safe(stable_rank(first)),
        "residual_second_stable_rank": safe(stable_rank(second)),
    }


def cross_features(left: torch.Tensor, right: torch.Tensor, prefix: str, max_lag: int) -> dict:
    max_lag = min(max_lag, left.numel() - 2)
    lags, values = lagged_cross_correlation(left.float(), right.float(), max_lag)
    peak = int(values.abs().argmax())
    zero = int((lags == 0).nonzero()[0])
    return {
        f"{prefix}_crosscorr_zero": safe(values[zero]),
        f"{prefix}_crosscorr_peak_abs": safe(values[peak].abs()),
        f"{prefix}_crosscorr_peak_lag": int(lags[peak]),
    }


def intermediate_target_logprob(model, state: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    last = model._last_token(state, mask)
    return target_log_probability(model.output(model.final_norm(last)), target)


@torch.inference_mode()
def run_checkpoint(checkpoint: Path, config: dict):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]
    model, splits, _, datasets = load_seed(model_config, checkpoint)
    seed = int(payload["seed"])
    device_name = resolve_device(model_config["training"].get("device", "auto"))
    device = torch.device(device_name)
    model.to(device).eval()
    loader = make_loader(datasets["test"], model_config, shuffle=False, seed=seed)
    records, repairs = [], []
    started = time.perf_counter()
    for batch in loader:
        ids, mask, targets = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["target"].to(device)
        native = model(ids, mask, capture=True)
        full_logprob = target_log_probability(native.logits, targets).cpu()
        correct = (native.logits.argmax(-1) == targets).cpu()
        interventions = {}
        for source in range(model.num_layers):
            interventions[(source, "skip_sa")] = model(ids, mask, skip_attention_layers={source}, capture=True)
            interventions[(source, "skip_ff")] = model(ids, mask, skip_ff_layers={source}, capture=True)
            interventions[(source, "skip_block")] = model(ids, mask, skip_layers={source}, capture=True)
        native_cpu = {
            "states": [x.cpu() for x in native.states],
            "candidate": [x.cpu() for x in native.candidates],
            "sa": [x.cpu() for x in native.attention_candidates],
            "ff": [x.cpu() for x in native.ff_candidates],
            "attention": [x.cpu() for x in native.attention],
        }
        for row, example_index in enumerate(batch["example_index"].tolist()):
            example = splits["test"][example_index]
            length = int(mask[row].sum())
            for layer in range(model.num_layers):
                series = {
                    "residual_norm": torch.linalg.vector_norm(native_cpu["states"][layer][row, :length].float(), dim=-1),
                    "block_update_norm": torch.linalg.vector_norm(native_cpu["candidate"][layer][row, :length].float(), dim=-1),
                    "sa_update_norm": torch.linalg.vector_norm(native_cpu["sa"][layer][row, :length].float(), dim=-1),
                    "ff_update_norm": torch.linalg.vector_norm(native_cpu["ff"][layer][row, :length].float(), dim=-1),
                    "attention_entropy": attention_metrics(native_cpu["attention"][layer][row, :, :length, :length])["attention_entropy"].mean(0),
                }
                record = {
                    "cycle": "B", "experiment": "B3", "revision": int(config["revision"]),
                    "run_id": f"cycleA-e1-baseline-seed-{seed}", "model_family": "tiny_custom_transformer",
                    "model_variant": "baseline", "task_family": example.intent, "example_id": example.example_id,
                    "layer": layer, "seed": seed, "full_correct": bool(correct[row]), "token_count": length,
                }
                for name, values in series.items():
                    record.update(scalar_features(values, name, config))
                record.update(vector_window_features(native_cpu["states"][layer][row, :length], int(config["capture"]["subspace_rank"])))
                record.update(cross_features(series["attention_entropy"], series["sa_update_norm"], "attention_entropy_sa_norm", int(config["capture"]["max_lag"])))
                record.update(cross_features(series["sa_update_norm"], series["ff_update_norm"], "sa_ff_norm", int(config["capture"]["max_lag"])))
                record.update(cross_features(series["residual_norm"], series["block_update_norm"], "residual_update_norm", int(config["capture"]["max_lag"])))
                records.append(record)

            for source in range(model.num_layers - 1):
                for intervention in config["capture"]["repair_interventions"]:
                    changed = interventions[(source, intervention)]
                    distances = [0.0]
                    task_gaps = [0.0]
                    for state_index in range(source + 1, model.num_layers + 1):
                        native_state = native.states[state_index][row, length - 1]
                        changed_state = changed.states[state_index][row, length - 1]
                        distances.append(safe(torch.linalg.vector_norm(native_state.float() - changed_state.float())))
                        native_lp = intermediate_target_logprob(model, native.states[state_index][row:row+1], mask[row:row+1], targets[row:row+1])
                        changed_lp = intermediate_target_logprob(model, changed.states[state_index][row:row+1], mask[row:row+1], targets[row:row+1])
                        task_gaps.append(safe((native_lp - changed_lp).abs()[0]))
                    vector_event = amplification_repair(
                        torch.tensor(distances), amplification_ratio=float(config["statistics"]["amplification_ratio"]),
                        repair_fraction=float(config["statistics"]["repair_fraction"]),
                    )
                    task_peak = max(task_gaps)
                    task_peak_index = task_gaps.index(task_peak)
                    task_repair = task_peak_index < len(task_gaps) - 1 and task_gaps[-1] <= task_peak * (1 - float(config["statistics"]["repair_fraction"]))
                    changed_lp_final = target_log_probability(changed.logits[row:row+1], targets[row:row+1]).cpu()[0]
                    repairs.append({
                        "cycle": "B", "experiment": "B3", "run_id": f"cycleA-e1-baseline-seed-{seed}",
                        "task_family": example.intent, "example_id": example.example_id, "source_layer": source,
                        "intervention": intervention, "seed": seed, "full_correct": bool(correct[row]),
                        "utility": safe(full_logprob[row] - changed_lp_final), "vector_repair": bool(vector_event["detected"]),
                        "vector_repair_score": float(vector_event["repair_score"]), "task_repair": bool(task_repair),
                        "task_repair_score": (task_peak - task_gaps[-1]) / max(task_peak, 1e-8),
                        "replicated_repair_event": bool(vector_event["detected"] and task_repair),
                        "initial_nonzero_divergence": distances[1], "peak_divergence": max(distances),
                        "final_divergence": distances[-1], "peak_task_gap": task_peak, "final_task_gap": task_gaps[-1],
                    })
    registry = {
        "run_id": f"cycleA-e1-baseline-seed-{seed}", "seed": seed, "checkpoint": str(checkpoint).replace("\\", "/"),
        "source_commit": current_commit(), "stability_rows": len(records), "repair_rows": len(repairs),
        "elapsed_seconds": time.perf_counter() - started, "device": device_name,
    }
    return registry, records, repairs


def correctness_summary(records: list[dict], config: dict) -> list[dict]:
    frame = pd.DataFrame(records)
    metadata = {"cycle", "experiment", "revision", "run_id", "model_family", "model_variant", "task_family", "example_id", "layer", "seed", "full_correct", "token_count"}
    metrics = [column for column in frame.columns if column not in metadata and not column.endswith("peak_lag")]
    rows = []
    threshold = float(config["statistics"]["minimum_standardized_correctness_effect"])
    for (task, layer), group in frame.groupby(["task_family", "layer"]):
        for metric in metrics:
            seed_effects = []
            for _, seed_group in group.groupby("seed"):
                correct = seed_group[seed_group.full_correct][metric]
                incorrect = seed_group[~seed_group.full_correct][metric]
                if len(correct) < 2 or len(incorrect) < 2:
                    continue
                pooled = math.sqrt((correct.var(ddof=1) + incorrect.var(ddof=1)) / 2)
                seed_effects.append((correct.mean() - incorrect.mean()) / max(pooled, 1e-8))
            if len(seed_effects) < 2:
                continue
            mean, low, high = confidence(seed_effects, seed=int(layer) + len(metric))
            rows.append({
                "task_family": task, "layer": int(layer), "metric": metric, "seed_n": len(seed_effects),
                "standardized_mean": mean, "median": float(pd.Series(seed_effects).median()),
                "std": float(pd.Series(seed_effects).std(ddof=1)), "ci95_low": low, "ci95_high": high,
                "replicated": bool((low > threshold and sum(x > threshold for x in seed_effects) >= 4) or (high < -threshold and sum(x < -threshold for x in seed_effects) >= 4)),
            })
    return rows


def repair_summary(repairs: list[dict], config: dict) -> list[dict]:
    frame = pd.DataFrame(repairs)
    rows = []
    threshold = float(config["statistics"]["repeated_repair_rate"])
    for keys, group in frame.groupby(["task_family", "source_layer", "intervention"]):
        task, source, intervention = keys
        seed_rates = group.groupby("seed")["replicated_repair_event"].mean().tolist()
        mean, low, high = confidence(seed_rates, seed=int(source) + len(intervention))
        rows.append({
            "task_family": task, "source_layer": int(source), "intervention": intervention,
            "seed_n": len(seed_rates), "repair_rate_mean": mean, "repair_rate_median": float(pd.Series(seed_rates).median()),
            "repair_rate_std": float(pd.Series(seed_rates).std(ddof=1)), "ci95_low": low, "ci95_high": high,
            "repeated": bool(low > threshold and sum(x > threshold for x in seed_rates) >= 4),
            "mean_utility": float(group.utility.mean()),
        })
    return rows


def plot(records: list[dict], repairs: list[dict], output: Path):
    frame = pd.DataFrame(records)
    repair = pd.DataFrame(repairs)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    for correct, label in ((True, "correct"), (False, "incorrect")):
        values = frame[frame.full_correct == correct].groupby("layer")["residual_norm_acf_lag1"].mean()
        axes[0].plot(values.index, values.values, marker="o", label=label)
    axes[0].set(title="Residual-norm token ACF(1)", xlabel="layer", ylabel="ACF")
    axes[0].legend()
    values = repair.groupby(["source_layer", "intervention"])["replicated_repair_event"].mean().unstack()
    values.plot(kind="bar", ax=axes[1])
    axes[1].set(title="Vector + task recovery event rate", xlabel="source layer", ylabel="rate")
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "b3_stability_repair.png", dpi=180); fig.savefig(figure_dir / "b3_stability_repair.pdf"); plt.close(fig)


def main():
    args = parse_args(); config_path = Path(args.config); config = read_yaml(config_path)
    output = Path(args.output or config["output"]["directory"]); output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "config_snapshot.yaml")
    metadata = RunMetadata.collect(
        run_id="cycle-b-b3-stability", model="tiny_residual_decoder", model_variant="baseline",
        dataset="synthetic_counterfactual_v1", seed=0, intervention="native_skip_sa_skip_ff_skip_block",
        context_length=56, batch_size=64, dtype="float32", device=resolve_device("auto"),
        hook_locations={"token_axis": "nonpadding token position", "depth_axis": "TinyOutput.states", "repair_state": "final-token residual after each later block"},
    )
    save_manifest(output / "manifest.json", metadata, config)
    checkpoints = sorted(Path().glob(config["checkpoint_glob"])); checkpoints = checkpoints[:args.max_checkpoints] if args.max_checkpoints else checkpoints
    registries, records, repairs = [], [], []
    log_lines = [f"B3 start checkpoints={len(checkpoints)} source_commit={current_commit()}"]
    for index, checkpoint in enumerate(checkpoints, 1):
        print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
        registry, run_records, run_repairs = run_checkpoint(checkpoint, config)
        registries.append(registry); records.extend(run_records); repairs.extend(run_repairs)
        log_lines.append(f"complete {registry['run_id']} seconds={registry['elapsed_seconds']:.3f}")
    correctness = correctness_summary(records, config); repair_cells = repair_summary(repairs, config)
    save_records_csv(output / "checkpoint_registry.csv", registries)
    save_records_parquet(output / "stability_records.parquet", records)
    save_records_csv(
        output / "correctness_contrasts.csv",
        correctness or [{"status": "insufficient_seeds_for_contrast"}],
    )
    save_records_parquet(output / "repair_events.parquet", repairs)
    save_records_csv(output / "repair_summary.csv", repair_cells)
    plot(records, repairs, output)
    record_frame = pd.DataFrame(records)
    repair_frame = pd.DataFrame(repairs)
    replicated_correctness = [row for row in correctness if row["replicated"]]
    key_depth_means = {
        metric: {
            str(int(layer)): float(group[metric].mean())
            for layer, group in record_frame.groupby("layer")
        }
        for metric in (
            "residual_norm_acf_lag1",
            "block_update_norm_acf_lag1",
            "attention_entropy_acf_lag1",
            "residual_covariance_drift",
            "residual_subspace_drift",
            "residual_eigenspectrum_drift",
            "residual_window_cka",
            "residual_first_stable_rank",
            "residual_second_stable_rank",
        )
    }
    summary = {
        "cycle": "B", "experiment": "B3", "revision": int(config["revision"]), "source_commit": current_commit(),
        "checkpoints": len(checkpoints), "stability_rows": len(records), "repair_rows": len(repairs),
        "replicated_correctness_cells": sum(row["replicated"] for row in correctness),
        "repeated_repair_cells": sum(row["repeated"] for row in repair_cells),
        "repair_event_rate": sum(row["replicated_repair_event"] for row in repairs) / len(repairs),
        "vector_repair_rate": float(repair_frame["vector_repair"].mean()),
        "task_gap_repair_rate": float(repair_frame["task_repair"].mean()),
        "replicated_correctness_by_task_layer": (
            {
                f"{task}-layer-{int(layer)}": int(len(group))
                for (task, layer), group in pd.DataFrame(replicated_correctness).groupby(
                    ["task_family", "layer"]
                )
            }
            if replicated_correctness
            else {}
        ),
        "replicated_correctness_metrics": [
            {
                "task_family": row["task_family"],
                "layer": int(row["layer"]),
                "metric": row["metric"],
                "standardized_mean": row["standardized_mean"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
            }
            for row in replicated_correctness
        ],
        "key_depth_means": key_depth_means,
        "interpretation": "Correctness contrasts are standardized seed-level exploratory tests; repair requires both vector recovery and reduced intermediate target-log-probability gap.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# B3 failure and null-result notes\n\n- Constant baseline gates have undefined correlation and are not analyzed.\n- Intermediate output-head target log probability is a task-relevant proxy, not a trained early-exit score.\n- Correctness contrasts are exploratory and multiplicity is not converted into a confirmatory claim.\n- Repair requires simultaneous vector and task-gap recovery; norm shrinkage alone is insufficient.\n",
        encoding="utf-8",
    )
    log_lines.append(f"B3 complete stability_rows={len(records)} repair_rows={len(repairs)}")
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
