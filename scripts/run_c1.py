"""Run Cycle C1 fixed-budget residual-gating learnability comparisons."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from gated_residuals.common.metrics import grad_norm
from gated_residuals.experiments import (
    build_model,
    confidence,
    current_commit,
    evaluate,
    make_loader,
    prepare_data,
    resolve_device,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/c1_learning_curves.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def threshold_label(threshold: float) -> str:
    return str(int(round(100 * threshold)))


@torch.inference_mode()
def update_diagnostics(model, batch, device: str) -> dict[str, float]:
    model.eval()
    out = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        capture=True,
    )
    candidate = torch.stack(
        [torch.linalg.vector_norm(value.float(), dim=-1).mean() for value in out.candidates]
    )
    effective = torch.stack(
        [torch.linalg.vector_norm(value.float(), dim=-1).mean() for value in out.effective_updates]
    )
    gates = torch.cat([value.float().reshape(-1) for value in out.gates])
    return {
        "candidate_update_norm": float(candidate.mean()),
        "effective_update_norm": float(effective.mean()),
        "effective_candidate_ratio": float(effective.mean() / candidate.mean().clamp_min(1e-12)),
        "applied_scale_mean": float(gates.mean()),
        "applied_scale_std": float(gates.std(unbiased=False)),
    }


def train_run(config: dict, variant: str, seed: int, run_dir: Path):
    seed_everything(seed)
    splits, vocabulary, datasets = prepare_data(config)
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config, len(vocabulary), variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    train_loader = make_loader(datasets["train"], config, shuffle=True, seed=seed)
    val_loader = make_loader(datasets["val"], config, shuffle=False, seed=seed)
    test_loader = make_loader(datasets["test"], config, shuffle=False, seed=seed)
    diagnostic_batch = next(iter(val_loader))
    eval_every = int(config["training"]["eval_every_steps"])
    max_grad = float(config["training"].get("max_grad_norm", 1.0))
    records = []
    global_step = tokens_seen = examples_seen = 0
    loss_total = grad_total = 0.0
    window_examples = window_steps = 0
    sync(device)
    started = time.perf_counter()

    initial = evaluate(model, val_loader, device)
    diagnostics = update_diagnostics(model, diagnostic_batch, device)
    records.append(
        {
            "variant": variant,
            "seed": seed,
            "epoch": 0,
            "global_step": 0,
            "tokens_seen": 0,
            "examples_seen": 0,
            "wall_seconds": 0.0,
            "train_loss": None,
            "gradient_norm": None,
            "parameter_update_norm": None,
            "val_loss": initial["loss"],
            "val_accuracy": initial["accuracy"],
            **diagnostics,
        }
    )

    epochs = int(config["training"]["epochs"])
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids, mask).logits, target)
            loss.backward()
            current_grad = grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            will_evaluate = (global_step + 1) % eval_every == 0
            before = None
            if will_evaluate:
                before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            global_step += 1
            batch_examples = target.numel()
            batch_tokens = int(mask.sum())
            examples_seen += batch_examples
            tokens_seen += batch_tokens
            loss_total += float(loss.detach()) * batch_examples
            grad_total += current_grad
            window_examples += batch_examples
            window_steps += 1
            is_final = epoch == epochs - 1 and global_step == epochs * len(train_loader)
            if will_evaluate or is_final:
                update_norm = None
                if before is not None:
                    squared = sum(
                        float((after.detach() - prior).float().square().sum())
                        for after, prior in zip(model.parameters(), before)
                    )
                    update_norm = math.sqrt(squared)
                validation = evaluate(model, val_loader, device)
                diagnostics = update_diagnostics(model, diagnostic_batch, device)
                sync(device)
                records.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "examples_seen": examples_seen,
                        "wall_seconds": time.perf_counter() - started,
                        "train_loss": loss_total / window_examples,
                        "gradient_norm": grad_total / window_steps,
                        "parameter_update_norm": update_norm,
                        "val_loss": validation["loss"],
                        "val_accuracy": validation["accuracy"],
                        **diagnostics,
                    }
                )
                loss_total = grad_total = 0.0
                window_examples = window_steps = 0
                model.train()

    final_test = evaluate(model, test_loader, device)
    sync(device)
    training_seconds = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocabulary": vocabulary.itos,
            "variant": variant,
            "seed": seed,
            "config": config,
            "selection": "fixed_budget_final_state",
        },
        checkpoint,
    )
    (run_dir / "history.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    save_records_parquet(run_dir / "learning_curve.parquet", records)
    metadata = RunMetadata.collect(
        run_id=f"c1-{variant}-seed-{seed}",
        model="tiny_residual_decoder",
        model_variant=variant,
        dataset="synthetic_counterfactual_v2_selection",
        seed=seed,
        context_length=int(config["data"]["max_length"]),
        batch_size=int(config["training"]["batch_size"]),
        gate_tensor_semantics=(
            "unconstrained learned layer scale" if variant == "static_scale" else
            "sigmoid residual multiplier" if "gated" in variant else "identity"
        ),
        dtype="float32",
        device=device,
    )
    save_manifest(run_dir / "manifest.json", metadata, config)
    registry = {
        "cycle": "C",
        "experiment": "C1",
        "revision": int(config["revision"]),
        "run_id": f"{variant}-seed-{seed}",
        "variant": variant,
        "seed": seed,
        "depth": model.num_layers,
        "width": model.width,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_steps": global_step,
        "tokens_seen": tokens_seen,
        "training_seconds": training_seconds,
        "final_val_loss": records[-1]["val_loss"],
        "final_val_accuracy": records[-1]["val_accuracy"],
        "final_test_loss": final_test["loss"],
        "final_test_accuracy": final_test["accuracy"],
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "device": device,
    }
    return records, registry


def summarize(curves: list[dict], registry: list[dict], config: dict):
    frame = pd.DataFrame(curves)
    thresholds = [float(value) for value in config["statistics"]["competence_thresholds"]]
    seed_rows = []
    registry_by_run = {row["run_id"]: row for row in registry}
    for (variant, seed), group in frame.groupby(["variant", "seed"]):
        group = group.sort_values("tokens_seen")
        maximum_tokens = float(group.tokens_seen.max())
        auc = float(np.trapz(group.val_accuracy, group.tokens_seen) / maximum_tokens)
        run = registry_by_run[f"{variant}-seed-{seed}"]
        row = {
            "variant": variant,
            "seed": int(seed),
            "learning_curve_auc": auc,
            "final_val_loss": float(group.iloc[-1].val_loss),
            "final_val_accuracy": float(group.iloc[-1].val_accuracy),
            "final_test_loss": run["final_test_loss"],
            "final_test_accuracy": run["final_test_accuracy"],
            "mean_gradient_norm": float(group.gradient_norm.dropna().mean()),
            "mean_parameter_update_norm": float(group.parameter_update_norm.dropna().mean()),
            "final_candidate_update_norm": float(group.iloc[-1].candidate_update_norm),
            "final_effective_update_norm": float(group.iloc[-1].effective_update_norm),
            "final_applied_scale_mean": float(group.iloc[-1].applied_scale_mean),
        }
        for threshold in thresholds:
            label = threshold_label(threshold)
            reached = group[group.val_accuracy >= threshold]
            first = None if reached.empty else reached.iloc[0]
            row[f"steps_to_{label}"] = None if first is None else int(first.global_step)
            row[f"tokens_to_{label}"] = None if first is None else int(first.tokens_seen)
            row[f"seconds_to_{label}"] = None if first is None else float(first.wall_seconds)
        seed_rows.append(row)

    summary_rows = []
    seeds = pd.DataFrame(seed_rows)
    metrics = [
        "learning_curve_auc",
        "final_val_loss",
        "final_val_accuracy",
        "final_test_loss",
        "final_test_accuracy",
        "mean_gradient_norm",
        "mean_parameter_update_norm",
        "final_candidate_update_norm",
        "final_effective_update_norm",
        "final_applied_scale_mean",
    ] + [
        f"{unit}_to_{threshold_label(threshold)}"
        for threshold in thresholds
        for unit in ("steps", "tokens", "seconds")
    ]
    for variant, group in seeds.groupby("variant"):
        row = {"variant": variant, "seed_n": len(group)}
        for metric in metrics:
            values = group[metric].dropna().astype(float).tolist()
            row[f"{metric}_reached_n"] = len(values)
            if values:
                mean, low, high = confidence(values, seed=len(variant) + len(metric))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_ci95_low"] = None
                row[f"{metric}_ci95_high"] = None
        summary_rows.append(row)

    comparisons = []
    baseline = seeds[seeds.variant == "baseline"].set_index("seed")
    for variant in ("static_scale", "gated", "sa_ff_gated"):
        other = seeds[seeds.variant == variant].set_index("seed")
        joined = baseline.join(other, lsuffix="_baseline", rsuffix="_variant")
        for metric in ("learning_curve_auc", "final_test_accuracy", "final_val_loss"):
            differences = (joined[f"{metric}_variant"] - joined[f"{metric}_baseline"]).tolist()
            mean, low, high = confidence(differences, seed=len(variant) + len(metric) + 101)
            comparisons.append(
                {
                    "variant": variant,
                    "reference": "baseline",
                    "metric": metric,
                    "paired_seed_n": len(differences),
                    "difference_mean": mean,
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                }
            )
    return seed_rows, summary_rows, comparisons


def plot_results(curves, seed_rows, output: Path):
    frame = pd.DataFrame(curves)
    seeds = pd.DataFrame(seed_rows)
    colors = {
        "baseline": "#222222",
        "static_scale": "#4c78a8",
        "gated": "#f58518",
        "sa_ff_gated": "#54a24b",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for variant, group in frame.groupby("variant"):
        pivot = group.pivot(index="tokens_seen", columns="seed", values="val_accuracy")
        axes[0].plot(pivot.index, pivot.mean(axis=1), label=variant, color=colors[variant])
        axes[0].fill_between(
            pivot.index,
            pivot.min(axis=1).to_numpy(),
            pivot.max(axis=1).to_numpy(),
            color=colors[variant],
            alpha=0.12,
        )
    axes[0].set(xlabel="Non-padding training tokens", ylabel="Validation accuracy", ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=8)
    order = list(colors)
    axes[1].boxplot(
        [seeds[seeds.variant == variant].learning_curve_auc for variant in order],
        labels=order,
        showmeans=True,
    )
    axes[1].set(ylabel="Normalized learning-curve AUC")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].boxplot(
        [seeds[seeds.variant == variant].final_test_accuracy for variant in order],
        labels=order,
        showmeans=True,
    )
    axes[2].set(ylabel="Final test accuracy", ylim=(0, 1.02))
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "c1_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(figures / "c1_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = read_yaml(args.config)
    output = Path(args.output or config["output"]["directory"])
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    curves, registry = [], []
    started = time.perf_counter()
    if args.postprocess_only:
        curves = pd.read_parquet(output / "learning_curves.parquet").to_dict("records")
        registry = pd.read_csv(output / "run_registry.csv").to_dict("records")
    else:
        for variant in config["variants"]:
            for seed in config["training"]["seeds"]:
                run_dir = output / "runs" / variant / f"seed-{seed}"
                history_path = run_dir / "history.json"
                if history_path.exists() and (run_dir / "model.pt").exists():
                    records = json.loads(history_path.read_text(encoding="utf-8"))
                    # A completed cached run also has its immutable registry row.
                    row = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                else:
                    records, row = train_run(config, variant, int(seed), run_dir)
                    (run_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                curves.extend(records)
                registry.append(row)
                print(json.dumps({"completed": row["run_id"], "test_accuracy": row["final_test_accuracy"], "seconds": row["training_seconds"]}), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    seed_rows, summary_rows, comparisons = summarize(curves, registry, config)
    save_records_parquet(output / "learning_curves.parquet", curves)
    save_records_csv(output / "run_registry.csv", registry)
    save_records_csv(output / "seed_summary.csv", seed_rows)
    save_records_csv(output / "variant_summary.csv", summary_rows)
    save_records_csv(output / "paired_comparisons.csv", comparisons)
    plot_results(curves, seed_rows, output)
    summary_frame = pd.DataFrame(summary_rows).set_index("variant")
    comparison_frame = pd.DataFrame(comparisons)
    result = {
        "cycle": "C",
        "experiment": "C1",
        "revision": int(config["revision"]),
        "source_commit": current_commit(),
        "runs": len(registry),
        "fixed_steps_per_run": int(registry[0]["training_steps"]),
        "fixed_tokens_per_run": sorted({int(row["tokens_seen"]) for row in registry}),
        "learning_curve_points": len(curves),
        "auc_by_variant": summary_frame.learning_curve_auc_mean.to_dict(),
        "final_test_accuracy_by_variant": summary_frame.final_test_accuracy_mean.to_dict(),
        "paired_auc_minus_baseline": {
            row.variant: row.difference_mean
            for row in comparison_frame[comparison_frame.metric == "learning_curve_auc"].itertuples()
        },
        "training_seconds_sum": sum(float(row["training_seconds"]) for row in registry),
        "stage_wall_seconds": None if args.postprocess_only else time.perf_counter() - started,
        "postprocess_only": bool(args.postprocess_only),
        "interpretation": "Learnability is assessed at a fixed optimization and token budget; terminal accuracy alone is not sufficient evidence.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "failure_null_notes.md").write_text(
        "# C1 failure and null-result notes\n\n"
        "- All variants receive identical data, optimizer, schedule, seed set, steps, and token budget.\n"
        "- Static scales are unconstrained learned layer constants initialized at 1; dynamic gates are sigmoid multipliers initialized at sigmoid(2).\n"
        "- Missing competence times are persisted as null and counted explicitly; they are not imputed.\n"
        "- Wall time includes periodic validation and diagnostics, so cross-variant timing is descriptive rather than a pure training-throughput benchmark.\n"
        "- Soft gates compute every SA/FF candidate and imply no realized FLOP saving.\n",
        encoding="utf-8",
    )
    metadata = RunMetadata.collect(
        run_id="cycle-c-c1",
        model="tiny_residual_decoder",
        model_variant="baseline+static_scale+gated+sa_ff_gated",
        dataset="synthetic_counterfactual_v2_selection",
        seed=0,
        context_length=int(config["data"]["max_length"]),
        batch_size=int(config["training"]["batch_size"]),
        dtype="float32",
        device=registry[0]["device"],
    )
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(
        f"C1 complete runs={len(registry)} training_seconds_sum={result['training_seconds_sum']:.3f} "
        f"stage_wall_seconds={result['stage_wall_seconds']} postprocess_only={args.postprocess_only}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
