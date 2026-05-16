from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ==============================
# Hardcoded configuration
# ==============================
MODEL_PATH = "/models/Qwen3-30B"
TOKENIZER_PATH = "/models/Qwen3-30B-tokenizer"

JSON_DATA_DIR = "data/instruction_json"
TEXT_DATA_DIR = "data/instruction_txt"

DATASET_CACHE_DIR = "outputs/mixed_instruction_dataset_qa_reasoning"
OUTPUT_DIR = "outputs/qwen3_lora_mixed_supervised_fsdp"

MAX_LENGTH = 1024
SEED = 42

# ~2000 pairs, 4 GPUs, small-to-medium effective batch to keep training stable.
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 4
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# LoRA tuned for instruction SFT on limited data
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

# FSDP is configured through accelerate config; these flags enable Trainer-side FSDP wrapping.
FSDP_MODE = "full_shard auto_wrap"
FSDP_LAYER_CLS_TO_WRAP = "Qwen3DecoderLayer,Qwen2DecoderLayer"


@dataclass
class QASample:
    question: str
    answer: str
    reasoning: str = ""


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_value(obj: dict, key_candidates: Iterable[str]) -> str:
    lowered_map = {k.lower().strip(): v for k, v in obj.items()}
    for key in key_candidates:
        key_l = key.lower().strip()
        if key_l in lowered_map and lowered_map[key_l] is not None:
            return str(lowered_map[key_l]).strip()
    return ""


def load_json_samples(json_dir: Path) -> list[QASample]:
    samples: list[QASample] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            payloads = [payload]
        elif isinstance(payload, list):
            payloads = [p for p in payload if isinstance(p, dict)]
        else:
            continue

        for item in payloads:
            q = _extract_value(item, ["question", "questions"])
            a = _extract_value(item, ["answers", "answer"])
            r = _extract_value(item, ["reasoning", "rationale", "analysis"])
            if q and a:
                samples.append(QASample(_normalize_text(q), _normalize_text(a), _normalize_text(r)))

    return samples


def load_text_samples(text_dir: Path) -> list[QASample]:
    samples: list[QASample] = []
    pattern = re.compile(
        r"(?is)\bquestion\s*[:\-]?\s*(.*?)\s*\banswer\s*[:\-]?\s*(.*)$",
        re.DOTALL,
    )

    for path in sorted(text_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = _normalize_text(raw)
        match = pattern.search(raw)
        if not match:
            continue
        q = _normalize_text(match.group(1))
        a = _normalize_text(match.group(2))
        if q and a:
            samples.append(QASample(question=q, answer=a))

    return samples


def format_chat_sample(sample: QASample) -> str:
    system = (
        "You are an expert AAA assistant. Provide accurate, grounded, and practical technical answers."
    )
    user_block = f"Question:\n{sample.question}"
    if sample.reasoning:
        assistant_block = f"Reasoning:\n{sample.reasoning}\n\nAnswer:\n{sample.answer}"
    else:
        assistant_block = f"Answer:\n{sample.answer}"

    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_block}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_block}\n"
        "<|im_end|>"
    )


def build_or_load_dataset(tokenizer) -> Dataset:
    cache_dir = Path(DATASET_CACHE_DIR)
    if cache_dir.exists():
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(str(cache_dir))

    json_samples = load_json_samples(Path(JSON_DATA_DIR))
    txt_samples = load_text_samples(Path(TEXT_DATA_DIR))
    all_samples = json_samples + txt_samples

    if not all_samples:
        raise ValueError("No valid training samples found in JSON/TXT data folders.")

    print(f"Loaded {len(json_samples)} JSON samples and {len(txt_samples)} TXT samples.")

    texts = [format_chat_sample(s) for s in all_samples]

    def tokenize_fn(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    ds = Dataset.from_dict({"text": texts})
    ds = ds.shuffle(seed=SEED)
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_dir))
    print(f"Saved tokenized dataset to {cache_dir}")
    return tokenized


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_or_load_dataset(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
    )

    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(
        len(dataset) / (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 4)
    )
    print(f"Approx steps/epoch on 4 GPUs: {steps_per_epoch}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fsdp=FSDP_MODE,
        fsdp_transformer_layer_cls_to_wrap=FSDP_LAYER_CLS_TO_WRAP,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
