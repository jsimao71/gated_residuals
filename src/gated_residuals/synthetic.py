"""Deterministic natural-language counterfactual families for Paper 1.

The latent intent labels returned by this module are measurement metadata.  They are
never inserted as task-ID tokens in the model input.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
INTENTS = ("maximum", "minimum", "sum_mod_10")
EXPANDED_INTENTS = (
    "maximum", "minimum", "sum_mod_10", "first", "middle", "last",
    "argmax_position", "argmin_position",
)
IDENTIFIABILITY = ("high", "medium", "low")
IDENTIFIABILITY_SCORE = {"high": 1.0, "medium": 0.5, "low": 0.0}

_GOAL_TEXT = {
    ("maximum", "high"): "The answer must be the largest value.",
    ("maximum", "medium"): "Choose the value that no listed value exceeds.",
    ("maximum", "low"): "In an ascending arrangement, return the item at the far right.",
    ("minimum", "high"): "The answer must be the smallest value.",
    ("minimum", "medium"): "Choose the value that exceeds no listed value.",
    ("minimum", "low"): "In an ascending arrangement, return the item at the far left.",
    ("sum_mod_10", "high"): "Add the values and answer with the final digit of the sum.",
    ("sum_mod_10", "medium"): "Combine all values; only the units place is wanted.",
    ("sum_mod_10", "low"): "After one full group of ten is repeatedly removed, return what remains.",
    ("first", "high"): "Return the first listed value.",
    ("first", "medium"): "Choose the value that appears before the other two.",
    ("first", "low"): "Read from left to right and return the leftmost item.",
    ("middle", "high"): "Return the middle listed value.",
    ("middle", "medium"): "Choose the value with one listed item on either side.",
    ("middle", "low"): "Keep the original order and return the central item.",
    ("last", "high"): "Return the last listed value.",
    ("last", "medium"): "Choose the value that appears after the other two.",
    ("last", "low"): "Read from left to right and return the rightmost item.",
    ("argmax_position", "high"): "Return first, second, or third for the position of the largest value.",
    ("argmax_position", "medium"): "Locate the greatest item and report its ordinal place.",
    ("argmax_position", "low"): "Find the rightmost item after sorting by value, but report where it was originally.",
    ("argmin_position", "high"): "Return first, second, or third for the position of the smallest value.",
    ("argmin_position", "medium"): "Locate the least item and report its ordinal place.",
    ("argmin_position", "low"): "Find the leftmost item after sorting by value, but report where it was originally.",
}
_PREFIXES = (
    "A short note contains these values:",
    "The report lists the following values:",
    "Read this small collection of values:",
)
_SUFFIXES = (
    "Reply using one number word.",
    "The response should contain exactly one number word.",
    "Give only the requested number word.",
)
_DISTRACTORS = (
    "The page margin is blue, which is irrelevant.",
    "A reviewer mentioned Tuesday, but that detail is unrelated.",
    "The values were copied neatly; this does not change the request.",
)


@dataclass(frozen=True)
class CounterfactualExample:
    example_id: str
    family_id: str
    split: str
    content: tuple[int, int, int]
    intent: str
    identifiability: str
    goal_identifiability: float
    distractor: bool
    prompt: str
    answer: str


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _content(seed: int, family_index: int) -> tuple[int, int, int]:
    rng = random.Random(_stable_seed(seed, "content", family_index))
    return tuple(rng.sample(range(10), 3))


def _answer(content: tuple[int, int, int], intent: str) -> str:
    if intent == "maximum":
        value = max(content)
    elif intent == "minimum":
        value = min(content)
    elif intent == "sum_mod_10":
        value = sum(content) % 10
    elif intent == "first":
        value = content[0]
    elif intent == "middle":
        value = content[1]
    elif intent == "last":
        value = content[2]
    elif intent == "argmax_position":
        return ("first", "second", "third")[content.index(max(content))]
    elif intent == "argmin_position":
        return ("first", "second", "third")[content.index(min(content))]
    else:
        raise ValueError(f"unknown intent: {intent}")
    return NUMBER_WORDS[value]


def generate_counterfactual_split(
    split: str,
    *,
    seed: int,
    families: int,
    family_offset: int,
    distractor_probability: float = 0.5,
    intents: tuple[str, ...] = INTENTS,
    identifiability_levels: tuple[str, ...] = IDENTIFIABILITY,
) -> list[CounterfactualExample]:
    """Generate content-disjoint families with every intent represented per content."""
    examples: list[CounterfactualExample] = []
    for local_index in range(families):
        family_index = family_offset + local_index
        family_id = f"family-{family_index:05d}"
        content = _content(seed, family_index)
        for intent_index, intent in enumerate(intents):
            identifiability = identifiability_levels[(family_index + intent_index) % len(identifiability_levels)]
            rng = random.Random(_stable_seed(seed, split, family_index, intent))
            distractor = rng.random() < distractor_probability
            value_text = " ".join(NUMBER_WORDS[value] for value in content)
            pieces = [
                _PREFIXES[rng.randrange(len(_PREFIXES))],
                value_text + ".",
                _GOAL_TEXT[(intent, identifiability)],
            ]
            if distractor:
                pieces.append(_DISTRACTORS[rng.randrange(len(_DISTRACTORS))])
            pieces.extend([_SUFFIXES[rng.randrange(len(_SUFFIXES))], "<answer>"])
            examples.append(
                CounterfactualExample(
                    example_id=f"{family_id}-{intent}",
                    family_id=family_id,
                    split=split,
                    content=content,
                    intent=intent,
                    identifiability=identifiability,
                    goal_identifiability=IDENTIFIABILITY_SCORE[identifiability],
                    distractor=distractor,
                    prompt=" ".join(pieces),
                    answer=_answer(content, intent),
                )
            )
    return examples


def build_splits(config: dict) -> dict[str, list[CounterfactualExample]]:
    data = config["data"]
    seed = int(data["generation_seed"])
    train_n, val_n, test_n = (int(data[name]) for name in ("train_families", "val_families", "test_families"))
    probability = float(data.get("distractor_probability", 0.5))
    intents = tuple(data.get("intents", INTENTS))
    identifiability_levels = tuple(data.get("identifiability_levels", IDENTIFIABILITY))
    if not intents or len(set(intents)) != len(intents) or not set(intents).issubset(EXPANDED_INTENTS):
        raise ValueError(f"data.intents must be a non-empty unique subset of {EXPANDED_INTENTS}")
    if not identifiability_levels or len(set(identifiability_levels)) != len(identifiability_levels) or not set(identifiability_levels).issubset(IDENTIFIABILITY):
        raise ValueError(f"data.identifiability_levels must be a non-empty unique subset of {IDENTIFIABILITY}")
    return {
        "train": generate_counterfactual_split("train", seed=seed, families=train_n, family_offset=0, distractor_probability=probability, intents=intents, identifiability_levels=identifiability_levels),
        "val": generate_counterfactual_split("val", seed=seed, families=val_n, family_offset=train_n, distractor_probability=probability, intents=intents, identifiability_levels=identifiability_levels),
        "test": generate_counterfactual_split("test", seed=seed, families=test_n, family_offset=train_n + val_n, distractor_probability=probability, intents=intents, identifiability_levels=identifiability_levels),
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"<[^>]+>|[a-z]+|\d+|[^\s]", text.lower())


class WordVocabulary:
    def __init__(self, texts: Iterable[str]):
        tokens = sorted({token for text in texts for token in tokenize(text)})
        self.itos = ["<pad>", "<unk>", *tokens]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenize(text)]

    def __len__(self) -> int:
        return len(self.itos)


class CounterfactualDataset(Dataset):
    def __init__(self, examples: list[CounterfactualExample], vocabulary: WordVocabulary, max_length: int):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        ids = self.vocabulary.encode(example.prompt)[-self.max_length :]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "target": torch.tensor(self.vocabulary.stoi[example.answer], dtype=torch.long),
            "example_index": index,
        }


def collate_counterfactual(batch: list[dict]) -> dict[str, torch.Tensor]:
    length = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.zeros((len(batch), length), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.bool)
    for row, item in enumerate(batch):
        size = item["input_ids"].numel()
        input_ids[row, :size] = item["input_ids"]
        attention_mask[row, :size] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": torch.stack([item["target"] for item in batch]),
        "example_index": torch.tensor([item["example_index"] for item in batch]),
    }
