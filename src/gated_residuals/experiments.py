"""Training and analysis primitives shared by Paper 1 experiment stages."""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .artifacts import RunMetadata, save_manifest, save_records_csv
from .common.config import load_yaml_config
from .residual_dynamics import bootstrap_mean_interval
from .synthetic import CounterfactualDataset, WordVocabulary, build_splits, collate_counterfactual
from .tiny_model import TinyResidualDecoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> str:
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def prepare_data(config: dict):
    splits = build_splits(config)
    all_text = [example.prompt for examples in splits.values() for example in examples]
    all_text.extend(example.answer for examples in splits.values() for example in examples)
    vocabulary = WordVocabulary(all_text)
    datasets = {
        split: CounterfactualDataset(examples, vocabulary, int(config["data"]["max_length"]))
        for split, examples in splits.items()
    }
    return splits, vocabulary, datasets


def make_loader(dataset, config: dict, *, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_counterfactual,
    )


def build_model(config: dict, vocab_size: int, variant: str) -> TinyResidualDecoder:
    return TinyResidualDecoder(vocab_size, variant=variant, **config["model"])


@torch.inference_mode()
def evaluate(model, loader, device: str, **forward_kwargs) -> dict:
    model.eval()
    losses, correct, total = 0.0, 0, 0
    probabilities, predictions, targets, indices = [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)
        logits = model(input_ids, attention_mask, **forward_kwargs).logits
        losses += float(F.cross_entropy(logits, target, reduction="sum"))
        prediction = logits.argmax(dim=-1)
        probability = logits.softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
        correct += int((prediction == target).sum())
        total += target.numel()
        probabilities.extend(probability.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        targets.extend(target.cpu().tolist())
        indices.extend(batch["example_index"].tolist())
    return {
        "loss": losses / total,
        "perplexity": math.exp(min(losses / total, 20.0)),
        "accuracy": correct / total,
        "probabilities": probabilities,
        "predictions": predictions,
        "targets": targets,
        "indices": indices,
    }


def train_seed(config: dict, variant: str, seed: int, output_dir: Path):
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
    best_loss = float("inf")
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        epoch_loss, count = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask).logits
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("max_grad_norm", 1.0)))
            optimizer.step()
            epoch_loss += float(loss.detach()) * target.numel()
            count += target.numel()
        validation = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / count, "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]})
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save({"model": model.state_dict(), "vocabulary": vocabulary.itos, "variant": variant, "seed": seed, "config": config}, checkpoint)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    metadata = RunMetadata.collect(
        run_id=f"{variant}-seed-{seed}", model="tiny_residual_decoder", model_variant=variant,
        dataset="synthetic_counterfactual_v1", seed=seed, context_length=int(config["data"]["max_length"]),
        batch_size=int(config["training"]["batch_size"]), dtype="float32", device=device,
    )
    save_manifest(output_dir / "manifest.json", metadata, config)
    return model, splits, vocabulary, datasets, {"training_seconds": time.perf_counter() - started, "best_val_loss": best_loss}


def load_seed(config: dict, checkpoint: Path):
    splits, vocabulary, datasets = prepare_data(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["vocabulary"] != vocabulary.itos:
        raise RuntimeError("checkpoint vocabulary does not match deterministic data vocabulary")
    model = build_model(config, len(vocabulary), payload["variant"])
    model.load_state_dict(payload["model"])
    return model, splits, vocabulary, datasets


def confidence(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value, value
    return bootstrap_mean_interval(torch.tensor(values), samples=2000, generator=torch.Generator().manual_seed(seed))


def write_summary(path: Path, rows: list[dict]) -> None:
    save_records_csv(path, rows)


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
