"""Run Cycle C2 goal-cue difficulty and minimum-depth sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gated_residuals.artifacts import RunMetadata, save_manifest, save_records_csv, save_records_parquet
from gated_residuals.common.config import read_yaml
from gated_residuals.experiments import confidence, current_commit
from run_c1 import train_run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c2_difficulty_depth.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def cell_config(config: dict, regime: str, depth: int) -> dict:
    result = copy.deepcopy(config)
    result["data"]["identifiability_levels"] = list(config["difficulty_regimes"][regime])
    result["model"]["layers"] = int(depth)
    return result


def add_cell_fields(records, row, *, sweep, regime, depth, reused_from=None):
    enriched = [
        {
            **record,
            "sweep": sweep,
            "difficulty_regime": regime,
            "depth": int(depth),
            "reused_from": reused_from,
        }
        for record in records
    ]
    registry = {
        **row,
        "sweep": sweep,
        "difficulty_regime": regime,
        "depth": int(depth),
        "reused_from": reused_from,
    }
    return enriched, registry


def load_c1_high(config: dict):
    source = Path(config["reuse"]["high_depth16"])
    curves = pd.read_parquet(source / "learning_curves.parquet")
    # Match C2's 20-step evaluation grid so trapezoidal AUC resolution is identical.
    maximum = curves.groupby(["variant", "seed"]).global_step.transform("max")
    curves = curves[(curves.global_step % int(config["training"]["eval_every_steps"]) == 0) | (curves.global_step == maximum)]
    registry = pd.read_csv(source / "run_registry.csv")
    all_curves, all_registry = [], []
    for row in registry.to_dict("records"):
        variant, seed = row["variant"], int(row["seed"])
        records = curves[(curves.variant == variant) & (curves.seed == seed)].to_dict("records")
        row["run_id"] = f"difficulty-high-depth16-{variant}-seed-{seed}"
        enriched, registered = add_cell_fields(
            records,
            row,
            sweep="difficulty",
            regime="high",
            depth=16,
            reused_from=str(source).replace("\\", "/"),
        )
        all_curves.extend(enriched)
        all_registry.append(registered)
    return all_curves, all_registry


def run_cell(config, output, *, sweep, regime, depth, variant, seed):
    run_id = f"{sweep}-{regime}-depth{depth}-{variant}-seed-{seed}"
    run_dir = output / "runs" / sweep / regime / f"depth-{depth}" / variant / f"seed-{seed}"
    if (run_dir / "history.json").exists() and (run_dir / "result.json").exists():
        records = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        row = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    else:
        records, row = train_run(
            cell_config(config, regime, depth),
            variant,
            int(seed),
            run_dir,
            run_id=run_id,
        )
        row.update({"sweep": sweep, "difficulty_regime": regime, "depth": int(depth), "reused_from": None})
        (run_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    enriched, registered = add_cell_fields(
        records,
        row,
        sweep=sweep,
        regime=regime,
        depth=depth,
    )
    return enriched, registered


def seed_summaries(curves, registry):
    frame = pd.DataFrame(curves)
    by_id = {row["run_id"]: row for row in registry}
    rows = []
    keys = ["sweep", "difficulty_regime", "depth", "variant", "seed"]
    for key, group in frame.groupby(keys):
        sweep, regime, depth, variant, seed = key
        group = group.sort_values("tokens_seen")
        run_id = f"{sweep}-{regime}-depth{int(depth)}-{variant}-seed-{int(seed)}"
        run = by_id[run_id]
        auc = float(np.trapz(group.val_accuracy, group.tokens_seen) / float(group.tokens_seen.max()))
        rows.append(
            {
                "sweep": sweep,
                "difficulty_regime": regime,
                "depth": int(depth),
                "variant": variant,
                "seed": int(seed),
                "learning_curve_auc": auc,
                "final_val_loss": float(group.iloc[-1].val_loss),
                "final_val_accuracy": float(group.iloc[-1].val_accuracy),
                "final_test_loss": float(run["final_test_loss"]),
                "final_test_accuracy": float(run["final_test_accuracy"]),
                "training_steps": int(run["training_steps"]),
                "tokens_seen": int(run["tokens_seen"]),
                "training_seconds": float(run["training_seconds"]),
            }
        )
    return rows


def aggregate(seed_rows, config):
    seeds = pd.DataFrame(seed_rows)
    threshold = float(config["statistics"]["competence_accuracy"])
    required = int(np.ceil(len(config["training"]["seeds"]) * float(config["statistics"]["minimum_seed_fraction"])))
    rows = []
    keys = ["sweep", "difficulty_regime", "depth", "variant"]
    metrics = ["learning_curve_auc", "final_val_loss", "final_val_accuracy", "final_test_loss", "final_test_accuracy", "training_seconds"]
    for key, group in seeds.groupby(keys):
        row = dict(zip(keys, key))
        row["seed_n"] = len(group)
        row["competent_seed_n"] = int((group.final_val_accuracy >= threshold).sum())
        row["reliably_competent"] = bool(
            row["competent_seed_n"] >= required and group.final_val_accuracy.mean() >= threshold
        )
        for metric in metrics:
            values = group[metric].tolist()
            mean, low, high = confidence(values, seed=len(metric) + len(str(key)))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def paired_comparisons(seed_rows):
    seeds = pd.DataFrame(seed_rows)
    rows = []
    keys = ["sweep", "difficulty_regime", "depth"]
    for key, cell in seeds.groupby(keys):
        baseline = cell[cell.variant == "baseline"].set_index("seed")
        for variant in ("static_scale", "gated", "sa_ff_gated"):
            other = cell[cell.variant == variant].set_index("seed")
            joined = baseline.join(other, lsuffix="_baseline", rsuffix="_variant")
            for metric in ("learning_curve_auc", "final_val_accuracy", "final_test_accuracy"):
                differences = (joined[f"{metric}_variant"] - joined[f"{metric}_baseline"]).tolist()
                mean, low, high = confidence(differences, seed=len(variant) + len(metric) + int(key[2]))
                rows.append(
                    {
                        "sweep": key[0],
                        "difficulty_regime": key[1],
                        "depth": int(key[2]),
                        "variant": variant,
                        "reference": "baseline",
                        "metric": metric,
                        "paired_seed_n": len(differences),
                        "difference_mean": mean,
                        "difference_ci95_low": low,
                        "difference_ci95_high": high,
                    }
                )
    return rows


def boundary_rows(summary_rows, config):
    summary = pd.DataFrame(summary_rows)
    results = []
    direct_order = list(config["difficulty_order"])
    for variant in config["variants"]:
        difficulty = summary[(summary.sweep == "difficulty") & (summary.variant == variant)]
        eligible = [
            regime for regime in direct_order
            if bool(difficulty[difficulty.difficulty_regime == regime].iloc[0].reliably_competent)
        ]
        depth = summary[
            (summary.sweep == "depth")
            & (summary.variant == variant)
            & (summary.difficulty_regime == config["depth_regime"])
        ].sort_values("depth")
        competent_depths = depth[depth.reliably_competent].depth.astype(int).tolist()
        tested_depths = depth.depth.astype(int).tolist()
        monotone_minima = [
            candidate
            for candidate in tested_depths
            if all(
                bool(depth[depth.depth == later].iloc[0].reliably_competent)
                for later in tested_depths
                if later >= candidate
            )
        ]
        mixed = difficulty[difficulty.difficulty_regime == "mixed"].iloc[0]
        results.append(
            {
                "variant": variant,
                "maximum_reliably_learned_directness_difficulty": eligible[-1] if eligible else None,
                "mixed_cue_reliably_competent_depth16": bool(mixed.reliably_competent),
                "lowest_qualifying_depth_mixed": min(competent_depths) if competent_depths else None,
                "monotone_minimum_depth_mixed": min(monotone_minima) if monotone_minima else None,
            }
        )
    return results


def plot(summary_rows, config, output):
    summary = pd.DataFrame(summary_rows)
    colors = {"baseline": "#222222", "static_scale": "#4c78a8", "gated": "#f58518", "sa_ff_gated": "#54a24b"}
    regimes = ["high", "medium", "low", "mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    difficulty = summary[summary.sweep == "difficulty"]
    for variant in config["variants"]:
        group = difficulty[difficulty.variant == variant].set_index("difficulty_regime").loc[regimes]
        axes[0].plot(regimes, group.final_val_accuracy_mean, marker="o", label=variant, color=colors[variant])
        axes[1].plot(regimes, group.learning_curve_auc_mean, marker="o", color=colors[variant])
    axes[0].axhline(float(config["statistics"]["competence_accuracy"]), color="gray", linestyle="--", linewidth=1)
    axes[0].set(ylabel="Final validation accuracy", ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel="Normalized learning-curve AUC")
    depth = summary[summary.sweep == "depth"]
    for variant in config["variants"]:
        group = depth[depth.variant == variant].sort_values("depth")
        axes[2].plot(group.depth, group.final_val_accuracy_mean, marker="o", label=variant, color=colors[variant])
    axes[2].axhline(float(config["statistics"]["competence_accuracy"]), color="gray", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Depth (mixed cues)", ylabel="Final validation accuracy", ylim=(0, 1.02), xticks=config["depths"])
    for axis in axes[:2]:
        axis.set_xlabel("Goal-cue regime")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "c2_difficulty_depth.pdf", bbox_inches="tight")
    fig.savefig(figures / "c2_difficulty_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = read_yaml(args.config)
    output = Path(args.output or config["output"]["directory"])
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    curves, registry = [], []
    started = time.perf_counter()
    if args.postprocess_only:
        curves = pd.read_parquet(output / "learning_curves.parquet").to_dict("records")
        registry = pd.read_csv(output / "run_registry.csv").to_dict("records")
    else:
        reused_curves, reused_registry = load_c1_high(config)
        curves.extend(reused_curves)
        registry.extend(reused_registry)
        cells = []
        for regime in ("medium", "low", "mixed"):
            cells.append(("difficulty", regime, 16))
        for depth in (4, 8, 12):
            cells.append(("depth", config["depth_regime"], depth))
        # The mixed/depth-16 cell is shared between the difficulty and depth summaries.
        tasks = [
            (sweep, regime, int(depth), variant, int(seed))
            for sweep, regime, depth in cells
            for variant in config["variants"]
            for seed in config["training"]["seeds"]
        ]
        for task_index, (sweep, regime, depth, variant, seed) in enumerate(tasks):
            if task_index % args.num_shards != args.shard_index:
                continue
            records, row = run_cell(
                config,
                output,
                sweep=sweep,
                regime=regime,
                depth=depth,
                variant=variant,
                seed=seed,
            )
            curves.extend(records)
            registry.append(row)
            print(json.dumps({"completed": row["run_id"], "val_accuracy": row["final_val_accuracy"], "test_accuracy": row["final_test_accuracy"], "seconds": row["training_seconds"]}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.train_only:
            print(json.dumps({"training_shard_complete": args.shard_index, "num_shards": args.num_shards}), flush=True)
            return
        mixed16_curves = [
            {**row, "sweep": "depth", "reused_from": "shared:difficulty-mixed-depth16"}
            for row in curves
            if row["sweep"] == "difficulty" and row["difficulty_regime"] == "mixed" and int(row["depth"]) == 16
        ]
        mixed16_registry = [
            {
                **row,
                "sweep": "depth",
                "run_id": row["run_id"].replace("difficulty-", "depth-", 1),
                "reused_from": "shared:difficulty-mixed-depth16",
            }
            for row in registry
            if row["sweep"] == "difficulty" and row["difficulty_regime"] == "mixed" and int(row["depth"]) == 16
        ]
        for row in mixed16_curves:
            row["sweep"] = "depth"
        curves.extend(mixed16_curves)
        registry.extend(mixed16_registry)

    seeds = seed_summaries(curves, registry)
    summaries = aggregate(seeds, config)
    comparisons = paired_comparisons(seeds)
    boundaries = boundary_rows(summaries, config)
    save_records_parquet(output / "learning_curves.parquet", curves)
    save_records_csv(output / "run_registry.csv", registry)
    save_records_csv(output / "seed_summary.csv", seeds)
    save_records_csv(output / "condition_summary.csv", summaries)
    save_records_csv(output / "paired_comparisons.csv", comparisons)
    save_records_csv(output / "boundary_summary.csv", boundaries)
    plot(summaries, config, output)
    result = {
        "cycle": "C",
        "experiment": "C2",
        "revision": int(config["revision"]),
        "source_commit": current_commit(),
        "new_runs": 0,
        "registry_rows_including_reuse": len(registry),
        "curve_rows": len(curves),
        "competence_accuracy": float(config["statistics"]["competence_accuracy"]),
        "boundaries": boundaries,
        "training_seconds_sum": sum(
            float(row["training_seconds"])
            for row in registry
            if pd.isna(row.get("reused_from"))
        ),
        "stage_wall_seconds": None if args.postprocess_only else time.perf_counter() - started,
        "postprocess_only": bool(args.postprocess_only),
        "interpretation": "Difficulty is goal-cue directness; mixed is a cue-diversity condition, not a fourth point on the directness scale. Depth qualification is validation-defined, and a minimum-depth claim additionally requires every tested deeper condition to qualify.",
    }
    # Avoid fragile NaN membership logic in JSON-facing counts.
    result["new_runs"] = sum(pd.isna(row.get("reused_from")) for row in registry)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# C2 failure and null-result notes\n\n"
        "- High-identifiability depth-16 cells reuse C1 fixed-final-state runs; their curves are downsampled to C2's 20-step grid.\n"
        "- Difficulty means goal-cue directness (high/medium/low identifiability). Mixed cues are reported separately because cue diversity is not ordinal directness.\n"
        "- The mixed-cue depth sweep was selected from the pre-existing B5 validation pilot, not C2 test behavior.\n"
        "- Reliable competence requires mean validation accuracy at least 0.90 and at least four of five seeds individually at least 0.90.\n"
        "- A lowest qualifying depth is not called a minimum necessary depth unless every tested deeper model also qualifies; the observed depth relation may be non-monotonic.\n"
        "- Every listed cell was fixed before C2 training; no depth or cue condition was added after frozen-test inspection.\n"
        "- Soft gates still evaluate all candidate writes and imply no avoided FLOPs.\n",
        encoding="utf-8",
    )
    metadata = RunMetadata.collect(
        run_id="cycle-c-c2",
        model="tiny_residual_decoder",
        model_variant="baseline+static_scale+gated+sa_ff_gated",
        dataset="synthetic_counterfactual_v2_goal_cues",
        seed=0,
        context_length=int(config["data"]["max_length"]),
        batch_size=int(config["training"]["batch_size"]),
        dtype="float32",
        device=registry[0]["device"],
    )
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(
        f"C2 complete new_runs={result['new_runs']} training_seconds_sum={result['training_seconds_sum']:.3f} "
        f"stage_wall_seconds={result['stage_wall_seconds']} postprocess_only={args.postprocess_only}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
