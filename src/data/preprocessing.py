"""
Dataset preprocessing for instruction fine-tuning.
Handles deduplication, quality filtering, tokenization, and packing.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError:
    Dataset = DatasetDict = load_dataset = None
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_instruction(example: dict) -> dict:
    text = INSTRUCTION_TEMPLATE.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )
    return {"text": text}


def deduplicate(examples: List[dict]) -> List[dict]:
    seen: set[str] = set()
    unique = []
    for ex in examples:
        h = hashlib.md5(ex["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    removed = len(examples) - len(unique)
    if removed:
        logger.info("Removed %d duplicate examples", removed)
    return unique


def quality_filter(example: dict, min_length: int = 50, max_length: int = 4000) -> bool:
    text = example.get("text", "")
    if len(text) < min_length or len(text) > max_length:
        return False
    # Reject examples with too many repeated characters (common in scraping artifacts)
    if len(set(text)) < 20:
        return False
    return True


def prepare_dataset(
    source_path: str,
    tokenizer_name: str,
    output_path: str,
    max_samples: int | None = None,
    train_split: float = 0.95,
) -> DatasetDict:
    logger.info("Loading dataset from %s", source_path)

    if source_path.startswith("s3://"):
        import boto3, tempfile, os
        s3 = boto3.client("s3")
        bucket, key = source_path[5:].split("/", 1)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            s3.download_file(bucket, key, tmp.name)
            raw = load_dataset("json", data_files=tmp.name, split="train")
    else:
        raw = load_dataset("json", data_files=source_path, split="train")

    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))

    logger.info("Raw dataset: %d examples", len(raw))

    # Format and filter
    formatted = raw.map(format_instruction, remove_columns=raw.column_names)
    filtered = formatted.filter(quality_filter)
    logger.info("After quality filter: %d examples", len(filtered))

    # Deduplicate
    deduped = Dataset.from_list(deduplicate(filtered.to_list()))
    logger.info("After dedup: %d examples", len(deduped))

    # Split
    split = deduped.train_test_split(test_size=1 - train_split, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    dataset.save_to_disk(output_path)
    logger.info("Saved to %s — train: %d, val: %d", output_path, len(dataset["train"]), len(dataset["validation"]))
    return dataset
