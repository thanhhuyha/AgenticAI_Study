"""Full-parameter Qwen3 SFT over three heterogeneous JSON schemas.

Run after configuring four-GPU FSDP with Accelerate:

    accelerate config
    accelerate launch scripts/sft_qwen3_full_fsdp_unified_three_json_types.py

Recommended Accelerate FSDP choices for a 30B model:
- FULL_SHARD, TRANSFORMER_BASED_WRAP, Qwen3DecoderLayer
- bf16, SHARDED_STATE_DICT, use_orig_params=true
- activation checkpointing=true, CPU RAM-efficient loading=true

FSDP topology and checkpoint behavior intentionally come from ``accelerate config``;
this script does not override them in TrainingArguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------------------------------------------------------
# Hardcoded standalone configuration
# -----------------------------------------------------------------------------
MODEL_WEIGHTS_DIR = "/models/Qwen3-30B"
TOKENIZER_DIR = "/models/Qwen3-30B-tokenizer"

# Each entry may be a .json/.jsonl file or a directory containing both formats.
DATA_SOURCES = [
    "data/sft/type1",
    "data/sft/type2",
    "data/sft/type3",
]
DATASET_CACHE_DIR = "outputs/qwen3_unified_three_types_cache"
OUTPUT_DIR = "outputs/qwen3_30b_full_sft_fsdp"

SYSTEM_PROMPT = (
    "You are an expert AAA programming assistant. Give accurate answers and "
    "produce valid AAA code when requested."
)
CODE_FENCE_LANGUAGE = "aaa"

MAX_LENGTH = 4096
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # effective batch = 64 on four GPUs
LEARNING_RATE = 2.0e-6
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 5
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
SEED = 42

# Raw reasoning / chain-of-thought is deliberately excluded by default. Turning
# this on trains the model to emit source rationale verbatim; only enable it if
# every rationale has been rewritten as a short, safe, user-facing explanation.
INCLUDE_BRIEF_RATIONALE = False

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalized_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize case, spaces, and hyphens in keys without changing values."""
    return {
        str(key).strip().lower().replace(" ", "_").replace("-", "_"): value
        for key, value in row.items()
    }


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Read object/list JSON or object-per-line JSONL."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield row
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}[{index}]")
            yield row
    else:
        raise ValueError(f"Expected JSON object or list in {path}")


def discover_data_files(sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"})
        else:
            raise FileNotFoundError(f"Data source not found or unsupported: {path}")
    files = sorted(set(files))
    if not files:
        raise ValueError("No .json or .jsonl training files were found.")
    return files


def rationale_prefix(row: dict[str, Any]) -> str:
    if not INCLUDE_BRIEF_RATIONALE:
        return ""
    rationale = text(row.get("brief_explanation") or row.get("reasoning"))
    return f"Brief explanation:\n{rationale}\n\n" if rationale else ""


def convert_row(raw_row: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Convert type 1/2/3 into one user/assistant representation."""
    row = normalized_fields(raw_row)
    question = text(row.get("question"))
    if not question:
        raise ValueError("Missing non-empty question/Question field")

    # Type 2 must be checked before generic answer-style records.
    type2_keys = {"choice_a", "choice_b", "choice_c", "choice_d", "final_choice"}
    if type2_keys.issubset(row):
        choices = "\n".join(
            f"{letter}. {text(row[f'choice_{letter.lower()}'])}" for letter in "ABCD"
        )
        user = f"{question}\n\nChoices:\n{choices}\n\nChoose the correct answer."
        final_choice = text(row["final_choice"])
        assistant = f"{rationale_prefix(row)}Final choice: {final_choice}"
        return {"user": user, "assistant": assistant}, "type2"

    # Type 3 supports the user's literal '<chain of thought>' spelling after key
    # normalization, but never trains on it unless a curated brief_explanation is
    # supplied and INCLUDE_BRIEF_RATIONALE is enabled.
    if "final_code" in row:
        final_code = text(row["final_code"])
        if not final_code:
            raise ValueError("Type 3 record has empty final_code")
        safe_rationale_row = dict(row)
        safe_rationale_row["reasoning"] = row.get("brief_explanation", "")
        assistant = (
            f"{rationale_prefix(safe_rationale_row)}Final code:\n"
            f"```{CODE_FENCE_LANGUAGE}\n{final_code}\n```"
        )
        return {"user": question, "assistant": assistant}, "type3"

    # Type 1: question + optional reasoning + answer.
    if "answer" in row:
        answer = text(row["answer"])
        if not answer:
            raise ValueError("Type 1 record has empty answer")
        assistant = f"{rationale_prefix(row)}Final answer:\n{answer}"
        return {"user": question, "assistant": assistant}, "type1"

    raise ValueError(f"Unknown schema with normalized fields: {sorted(row)}")


def cache_fingerprint(files: list[Path], tokenizer) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(str(TOKENIZER_DIR).encode())
    digest.update(str(MAX_LENGTH).encode())
    digest.update(str(INCLUDE_BRIEF_RATIONALE).encode())
    digest.update(str(getattr(tokenizer, "chat_template", None)).encode())
    return digest.hexdigest()


def tokenize_example(example: dict[str, str], tokenizer) -> dict[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    prompt_messages = messages[:-1]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)

    labels = list(full["input_ids"])
    prompt_length = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_length] = [-100] * prompt_length

    # Drop fully truncated samples: they contain no assistant target tokens.
    if not any(label != -100 for label in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def build_or_load_dataset(tokenizer) -> Dataset:
    files = discover_data_files(DATA_SOURCES)
    fingerprint = cache_fingerprint(files, tokenizer)
    cache_dir = Path(DATASET_CACHE_DIR) / fingerprint
    if cache_dir.exists():
        print(f"Loading matching cached dataset from {cache_dir}")
        return load_from_disk(str(cache_dir))

    examples: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    failures: list[str] = []
    for path in files:
        for index, row in enumerate(iter_json_rows(path)):
            try:
                example, record_type = convert_row(row)
                examples.append(example)
                counts[record_type] += 1
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"{path} row {index}: {exc}")

    if not examples:
        raise ValueError("No valid examples were produced.\n" + "\n".join(failures[:20]))

    print(f"Unified {len(examples)} examples: {dict(counts)}; skipped {len(failures)} invalid rows")
    dataset = Dataset.from_list(examples).shuffle(seed=SEED)
    dataset = dataset.map(
        lambda batch: tokenize_example(batch, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking prompt tokens",
    )
    dataset = dataset.filter(lambda row: len(row["input_ids"]) > 0, desc="Removing fully truncated examples")

    # Keep only the newest cache generated by this standalone script.
    cache_root = Path(DATASET_CACHE_DIR)
    if cache_root.exists():
        for old_cache in cache_root.iterdir():
            if old_cache.is_dir() and old_cache != cache_dir:
                shutil.rmtree(old_cache)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    return dataset


def main() -> None:
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = build_or_load_dataset(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_WEIGHTS_DIR,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # Do not use device_map="auto" for distributed FSDP training.
    )
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_safetensors=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        # FSDP and activation checkpointing are read from accelerate config.
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
