"""Run Cycle B7 on cached Qwen checkpoints without overstating the gated comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gated_residuals.adapters import (
    Qwen3AttentionAdapter,
    Qwen3ResidualIntervention,
    assert_native_parity,
)
from gated_residuals.artifacts import RunMetadata, save_manifest
from gated_residuals.attention_dilution import attention_metrics
from gated_residuals.common.config import read_yaml
from gated_residuals.residual_dynamics import (
    bootstrap_mean_interval,
    pairwise_layer_matrices,
    sa_ff_geometry,
    update_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b7_qwen.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite B7 metric")
    return result


def examples() -> list[dict[str, str]]:
    groups = {
        "ascending": [
            "1 2 3 4 5 6 7 8 9 10",
            "11 12 13 14 15 16 17 18 19 20",
            "21 22 23 24 25 26 27 28 29 30",
            "31 32 33 34 35 36 37 38 39 40",
        ],
        "descending": [
            "10 9 8 7 6 5 4 3 2 1",
            "20 19 18 17 16 15 14 13 12 11",
            "30 29 28 27 26 25 24 23 22 21",
            "40 39 38 37 36 35 34 33 32 31",
        ],
        "alternating": [
            "red blue red blue red blue red blue red blue",
            "up down up down up down up down up down",
            "left right left right left right left right",
            "yes no yes no yes no yes no yes no",
        ],
        "trigram": [
            "one two three one two three one two three one two three",
            "alpha beta gamma alpha beta gamma alpha beta gamma",
            "spring summer autumn spring summer autumn spring summer autumn",
            "small medium large small medium large small medium large",
        ],
        "week_cycle": [
            "Monday Tuesday Wednesday Thursday Friday Saturday Sunday Monday Tuesday",
            "Tuesday Wednesday Thursday Friday Saturday Sunday Monday Tuesday Wednesday",
            "Wednesday Thursday Friday Saturday Sunday Monday Tuesday Wednesday Thursday",
            "Thursday Friday Saturday Sunday Monday Tuesday Wednesday Thursday Friday",
        ],
        "constant": [
            "echo echo echo echo echo echo echo echo echo echo",
            "same same same same same same same same same same",
            "again again again again again again again again again",
            "repeat repeat repeat repeat repeat repeat repeat repeat",
        ],
    }
    return [
        {"example_id": f"{family}-{index}", "family": family, "text": text}
        for family, texts in groups.items()
        for index, text in enumerate(texts)
    ]


def _encode(tokenizer, records: list[dict[str, str]], max_length: int) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(
        [row["text"] for row in records],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.inference_mode()
def quality(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    count = encoded["input_ids"].shape[0]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ids = encoded["input_ids"][start:stop].to(device)
        mask = encoded["attention_mask"][start:stop].to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        losses = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            reduction="none",
        ).view(ids.shape[0], -1)
        valid = mask[:, 1:].float()
        values.append((-(losses * valid).sum(1) / valid.sum(1).clamp_min(1)).cpu())
    return torch.cat(values)


def _resource_audit(config: dict) -> dict:
    standard = Path(config["standard_model"]["local_snapshot"])
    gated = Path(config["gated_release"]["gated_snapshot"])
    baseline = Path(config["gated_release"]["baseline_snapshot"])
    gated_weights = gated / "pytorch_model.bin"
    state = torch.load(gated_weights, map_location="cpu", weights_only=True, mmap=True)
    gated_parameters = sum(value.numel() for value in state.values())
    gated_dtypes = sorted({str(value.dtype) for value in state.values()})
    gated_parameter_bytes = sum(value.numel() * value.element_size() for value in state.values())
    del state
    try:
        import psutil

        memory = psutil.virtual_memory()
        host_available = int(memory.available)
        host_total = int(memory.total)
    except ImportError:
        host_available = None
        host_total = None
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_free, device_total = torch.cuda.mem_get_info()
    else:
        device_free, device_total = 0, 0
    modeling = gated / "modeling_qwen3.py"
    source = modeling.read_text(encoding="utf-8")
    static_gate_verified = all(
        fragment in source
        for fragment in (
            "query_states, gate_score = torch.split",
            "gate_score = gate_score.reshape(bsz, q_len, -1, 1)",
            "attn_output = attn_output * torch.sigmoid(gate_score)",
            "attn_output = self.o_proj(attn_output)",
        )
    )
    reserve_host = int(config["resource_gate"]["reserve_host_bytes"])
    reserve_device = int(config["resource_gate"]["reserve_device_bytes"])
    host_fit = host_available is not None and gated_parameter_bytes + reserve_host <= host_available
    device_fit = gated_parameter_bytes + reserve_device <= device_free
    return {
        "captured_before_standard_model_load": True,
        "host_total_bytes": host_total,
        "host_available_bytes": host_available,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_total_bytes": int(device_total),
        "device_free_bytes": int(device_free),
        "disk_free_bytes": shutil.disk_usage(standard).free,
        "standard_checkpoint_cached": (standard / "model.safetensors").exists(),
        "standard_on_disk_bytes": (standard / "model.safetensors").stat().st_size,
        "standard_revision": config["standard_model"]["revision"],
        "gated_checkpoint_cached": gated_weights.exists(),
        "gated_on_disk_bytes": gated_weights.stat().st_size,
        "gated_named_parameter_count": gated_parameters,
        "gated_parameter_bytes": gated_parameter_bytes,
        "gated_dtypes": gated_dtypes,
        "baseline_config_cached": (baseline / "config.json").exists(),
        "baseline_weights_cached": (baseline / "pytorch_model.bin").exists(),
        "static_exact_revision_gate_sequence_verified": static_gate_verified,
        "modeling_sha256": _sha256(modeling),
        "configuration_sha256": _sha256(gated / "configuration_qwen3.py"),
        "host_fit_with_reserve": host_fit,
        "device_fit_with_reserve": device_fit,
        "exact_gated_load_eligible": bool(host_fit or device_fit),
        "known_exact_load_attempts": read_yaml("configs/e7_transfer.yaml")["load_attempt"],
        "new_download_bytes": 0,
    }


@torch.inference_mode()
def capture_atlas(model, tokenizer, records, config, device):
    batch_size = int(config["probe"]["capture_batch_size"])
    max_length = int(config["probe"]["max_length"])
    topk = int(config["probe"]["attention_topk"])
    geometry_rows: list[dict] = []
    attention_rows: list[dict] = []
    final_states = [[] for _ in range(len(model.model.layers))]
    final_updates = [[] for _ in range(len(model.model.layers))]
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        encoded = _encode(tokenizer, batch_records, max_length)
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        adapter = Qwen3AttentionAdapter(model, detach_to_cpu=True)
        with adapter:
            model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                output_attentions=True,
            )
        for row, record in enumerate(batch_records):
            length = int(mask[row].sum())
            for layer, capture in enumerate(adapter.captures()):
                residual = capture.residual_input[row, :length]
                attention_update = capture.candidate_update[row, :length]
                ff_update = torch.stack(
                    [adapter.ff_candidate_update(layer, token)[row] for token in range(length)]
                )
                residual_post = torch.stack(
                    [adapter.residual_post(layer, token)[row] for token in range(length)]
                )
                block_update = residual_post - residual
                sublayer = sa_ff_geometry(residual, attention_update, ff_update)
                block = update_geometry(residual, block_update)
                weights = capture.attention_weights[row, :, :length, :length]
                concentration = attention_metrics(weights, topk=topk)
                head_norm = torch.linalg.vector_norm(
                    capture.candidate_heads[row, :length].float(), dim=-1
                ).transpose(0, 1)
                final_states[layer].append(residual[-1])
                final_updates[layer].append(block_update[-1])
                for token in range(length):
                    row_values = {
                        "cycle": "B",
                        "experiment": "B7",
                        "revision": int(config["revision"]),
                        "model_family": "Qwen3",
                        "model": config["standard_model"]["repository"],
                        "model_revision": config["standard_model"]["revision"],
                        "model_variant": "standard_0.6B",
                        "task_family": record["family"],
                        "example_id": record["example_id"],
                        "layer": layer,
                        "token_index": token,
                        "is_final_token": token == length - 1,
                    }
                    for name, values in sublayer.items():
                        row_values[name] = _finite(values[token])
                    for name, values in block.items():
                        row_values[f"block_{name}"] = _finite(values[token])
                    geometry_rows.append(row_values)
                    for head in range(weights.shape[0]):
                        attention_rows.append(
                            {
                                "cycle": "B",
                                "experiment": "B7",
                                "revision": int(config["revision"]),
                                "model_family": "Qwen3",
                                "model_variant": "standard_0.6B",
                                "task_family": record["family"],
                                "example_id": record["example_id"],
                                "layer": layer,
                                "head": head,
                                "token_index": token,
                                "attention_entropy": _finite(
                                    concentration["attention_entropy"][head, token]
                                ),
                                "attention_normalized_entropy": _finite(
                                    concentration["attention_normalized_entropy"][head, token]
                                ),
                                "attention_effective_support": _finite(
                                    concentration["attention_effective_support"][head, token]
                                ),
                                "attention_top1_mass": _finite(
                                    concentration["attention_top1_mass"][head, token]
                                ),
                                "attention_topk_mass": _finite(
                                    concentration["attention_topk_mass"][head, token]
                                ),
                                "attention_sink_mass": _finite(
                                    concentration["attention_sink_mass"][head, token]
                                ),
                                "head_output_norm": _finite(head_norm[head, token]),
                            }
                        )
    states = torch.stack([torch.stack(values) for values in final_states])
    updates = torch.stack([torch.stack(values) for values in final_updates])
    matrices = pairwise_layer_matrices(states, updates)
    matrix_rows = [
        {
            "cycle": "B",
            "experiment": "B7",
            "model_family": "Qwen3",
            "matrix": name,
            "layer_i": left,
            "layer_j": right,
            "value": _finite(matrix[left, right]),
        }
        for name, matrix in matrices.items()
        for left in range(matrix.shape[0])
        for right in range(matrix.shape[1])
    ]
    return geometry_rows, attention_rows, matrix_rows


def _summarize(config, geometry_rows, attention_rows, causal_rows):
    geometry = pd.DataFrame(geometry_rows)
    attention = pd.DataFrame(attention_rows)
    causal = pd.DataFrame(causal_rows)
    layer = geometry.groupby("layer").agg(
        attention_relative_update_norm=("attention_relative_update_norm", "mean"),
        ff_relative_update_norm=("ff_relative_update_norm", "mean"),
        attention_ff_cosine=("attention_ff_cosine", "mean"),
        attention_ff_cancellation=("attention_ff_cancellation", "mean"),
        block_relative_update_norm=("block_relative_update_norm", "mean"),
    ).reset_index()
    attn_layer = attention.groupby("layer").agg(
        attention_entropy=("attention_entropy", "mean"),
        attention_top1_mass=("attention_top1_mass", "mean"),
        attention_sink_mass=("attention_sink_mass", "mean"),
        head_output_norm=("head_output_norm", "mean"),
    ).reset_index()
    layer = layer.merge(attn_layer, on="layer")
    generator = torch.Generator().manual_seed(int(config["probe"]["seed"]))
    utility_summary = []
    for (mode, layer_index), group in causal.groupby(["mode", "layer"]):
        values = torch.tensor(group["utility"].to_numpy())
        mean, low, high = bootstrap_mean_interval(
            values,
            confidence=float(config["probe"]["confidence"]),
            samples=int(config["probe"]["bootstrap_samples"]),
            generator=generator,
        )
        utility_summary.append(
            {
                "mode": mode,
                "layer": int(layer_index),
                "utility_mean": mean,
                "utility_ci95_low": low,
                "utility_ci95_high": high,
                "utility_median": float(values.median()),
                "utility_std": float(values.std(unbiased=True)),
                "n": int(values.numel()),
            }
        )
    utilities = pd.DataFrame(utility_summary)
    for mode, label in (("skip_attention", "sa"), ("skip_ff", "ff"), ("skip_block", "block")):
        selected = utilities[utilities["mode"] == mode][["layer", "utility_mean"]]
        layer = layer.merge(selected.rename(columns={"utility_mean": f"{label}_utility"}), on="layer")
    return geometry, attention, causal, layer, utilities


def _plot(output: Path, layer: pd.DataFrame, attention: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    axes[0].plot(layer["layer"], layer["attention_relative_update_norm"], label="SA")
    axes[0].plot(layer["layer"], layer["ff_relative_update_norm"], label="FF")
    axes[0].set(xlabel="Layer", ylabel="Mean relative update norm")
    axes[0].legend(frameon=False)
    axes[1].plot(layer["layer"], layer["sa_utility"], label="skip SA")
    axes[1].plot(layer["layer"], layer["ff_utility"], label="skip FF")
    axes[1].plot(layer["layer"], layer["block_utility"], label="skip block")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Layer", ylabel="Mean causal utility (nats/token)")
    axes[1].legend(frameon=False, fontsize=8)
    heat = attention.groupby(["layer", "token_index"])["attention_normalized_entropy"].mean().unstack()
    image = axes[2].imshow(heat.to_numpy(), aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[2].set(xlabel="Token index", ylabel="Layer", title="Normalized attention entropy")
    figure.colorbar(image, ax=axes[2], fraction=.046)
    figure.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "b7_qwen.pdf", bbox_inches="tight")
    figure.savefig(figures / "b7_qwen.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    output = Path(args.output or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "config_snapshot.yaml")
    started = time.perf_counter()
    resource = _resource_audit(config)
    records = examples()
    snapshot = Path(config["standard_model"]["local_snapshot"])
    if not resource["standard_checkpoint_cached"]:
        raise RuntimeError("pinned standard Qwen checkpoint is not cached")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=config["standard_model"]["attention_implementation"],
        low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = _encode(tokenizer, records, int(config["probe"]["max_length"]))
    parity_inputs = {
        "input_ids": encoded["input_ids"][:1].to(device),
        "attention_mask": encoded["attention_mask"][:1].to(device),
        "use_cache": False,
        "output_attentions": True,
    }
    parity_adapter = Qwen3AttentionAdapter(model, detach_to_cpu=True)
    parity_error = assert_native_parity(model, parity_adapter, parity_inputs, atol=0, rtol=0)
    if parity_error != 0:
        raise RuntimeError("standard Qwen instrumentation failed exact parity")
    geometry_rows, attention_rows, matrix_rows = capture_atlas(
        model, tokenizer, records, config, device
    )
    native = quality(
        model,
        encoded,
        batch_size=int(config["probe"]["intervention_batch_size"]),
        device=device,
    )
    causal_rows = []
    for layer in range(len(model.model.layers)):
        for mode in ("skip_attention", "skip_ff", "skip_block"):
            with Qwen3ResidualIntervention(model, layer=layer, mode=mode):
                skipped = quality(
                    model,
                    encoded,
                    batch_size=int(config["probe"]["intervention_batch_size"]),
                    device=device,
                )
            utility = native - skipped
            for index, record in enumerate(records):
                causal_rows.append(
                    {
                        "cycle": "B",
                        "experiment": "B7",
                        "revision": int(config["revision"]),
                        "model_family": "Qwen3",
                        "model": config["standard_model"]["repository"],
                        "model_revision": config["standard_model"]["revision"],
                        "model_variant": "standard_0.6B",
                        "example_id": record["example_id"],
                        "task_family": record["family"],
                        "layer": layer,
                        "mode": mode,
                        "native_quality": _finite(native[index]),
                        "skipped_quality": _finite(skipped[index]),
                        "utility": _finite(utility[index]),
                    }
                )
    geometry, attention, causal, layer, utilities = _summarize(
        config, geometry_rows, attention_rows, causal_rows
    )
    geometry.to_parquet(output / "residual_records.parquet", index=False)
    attention.to_parquet(output / "attention_records.parquet", index=False)
    causal.to_parquet(output / "causal_records.parquet", index=False)
    pd.DataFrame(matrix_rows).to_parquet(output / "layer_matrices.parquet", index=False)
    layer.to_csv(output / "layer_summary.csv", index=False)
    utilities.to_csv(output / "causal_summary.csv", index=False)
    _plot(output, layer, attention)
    correlations = {
        "attention_entropy_vs_sa_utility_spearman": float(
            layer["attention_entropy"].corr(layer["sa_utility"], method="spearman")
        ),
        "sa_norm_vs_sa_utility_spearman": float(
            layer["attention_relative_update_norm"].corr(layer["sa_utility"], method="spearman")
        ),
        "ff_norm_vs_ff_utility_spearman": float(
            layer["ff_relative_update_norm"].corr(layer["ff_utility"], method="spearman")
        ),
    }
    mode_summary = {
        mode: {
            "mean_utility": float(group["utility_mean"].mean()),
            "negative_layer_fraction": float((group["utility_mean"] < 0).mean()),
            "minimum_layer": int(group.loc[group["utility_mean"].idxmin(), "layer"]),
            "minimum_utility": float(group["utility_mean"].min()),
            "maximum_layer": int(group.loc[group["utility_mean"].idxmax(), "layer"]),
            "maximum_utility": float(group["utility_mean"].max()),
        }
        for mode, group in utilities.groupby("mode")
    }
    summary = {
        "cycle": "B",
        "experiment": "B7",
        "revision": int(config["revision"]),
        "status": "standard_qwen_complete_exact_gated_resource_stop",
        "standard_model": {
            "repository": config["standard_model"]["repository"],
            "revision": config["standard_model"]["revision"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "runtime_dtype": str(next(model.parameters()).dtype),
            "layers": len(model.model.layers),
            "native_parity_max_logit_error": parity_error,
            "examples": len(records),
            "native_mean_log_probability": float(native.mean()),
            "residual_rows": len(geometry),
            "attention_rows": len(attention),
            "causal_rows": len(causal),
            "matrix_rows": len(matrix_rows),
        },
        "standard_qwen_causal_utility": mode_summary,
        "standard_qwen_correlations": correlations,
        "gated_release": {
            "repository": config["gated_release"]["repository"],
            "revision": config["gated_release"]["revision"],
            "static_gate_sequence_verified": resource[
                "static_exact_revision_gate_sequence_verified"
            ],
            "real_native_parity_completed": False,
            "examples_evaluated": 0,
            "gate_metrics_available": False,
            "reason": "exact checkpoint exceeds measured host/device headroom and prior exact loads failed",
        },
        "resource_audit": resource,
        "b8_b9_gate_open": False,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "The common probe is numerically valid on a real small standard Qwen model. "
            "The matched released baseline/headwise comparison remains unevaluated, so B7 "
            "does not establish gated-Qwen transfer and does not open B8/B9."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    notes = (
        "# B7 failure and null-result notes\n\n"
        "The cached Qwen3-0.6B standard model completed exact native-forward parity, the "
        "residual/SA/FF atlas, attention metrics, and all layerwise causal ablations. The "
        "exact pinned headwise gated release was not retried because its measured parameter "
        "storage exceeds current RAM/VRAM headroom and the same checkpoint previously failed "
        "ordinary CPU materialization and two direct GPU placements. No baseline release "
        "weights were downloaded, no gated examples were evaluated, and no gated-attention "
        "behavioral claim is made. B8/B9 remain closed because B7 did not work end-to-end.\n"
    )
    (output / "failure_null_notes.md").write_text(notes, encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id="cycle-b-b7-qwen",
        model=config["standard_model"]["repository"],
        model_variant="standard_0.6B_plus_gated_resource_audit",
        dataset="deterministic_predictable_sequence_probe",
        seed=int(config["probe"]["seed"]),
        intervention="native+skip_attention+skip_ff+skip_block",
        model_revision=config["standard_model"]["revision"],
        tokenizer=config["standard_model"]["repository"],
        context_length=int(config["probe"]["max_length"]),
        batch_size=int(config["probe"]["intervention_batch_size"]),
        hook_locations={
            "residual_pre": "decoder_layer_input",
            "attention_candidate": "attention_o_proj_output",
            "residual_after_attention": "residual_pre+attention_output",
            "ff_candidate": "mlp_output",
            "residual_post": "decoder_layer_output",
        },
        dtype=str(next(model.parameters()).dtype),
        device=str(device),
    )
    save_manifest(output / "manifest.json", metadata, config)
    (output / "run.log").write_text(
        f"B7 standard Qwen complete; exact gated resource stop; elapsed_seconds={summary['elapsed_seconds']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
