from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Hardcoded params
MODEL_PATH = "/path/to/local/Qwen3-30B"
QA_JSON_DIR = Path("outputs/generated_instruction_qa")
DATASET_CACHE_DIR = Path("outputs/generated_json_qa_dataset")
OUTPUT_DIR = Path("outputs/qwen3_generated_json_qa_ft")

USE_LORA = True  # Set False for full-parameter fine-tuning
MAX_LENGTH = 1024

LEARNING_RATE = 3e-5 if USE_LORA else 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
MAX_STEPS = 1500
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def format_chat(question: str, answer: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful AAA assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{answer}\n"
        "<|im_end|>"
    )


def build_or_load_dataset(tokenizer: AutoTokenizer) -> Dataset:
    if DATASET_CACHE_DIR.exists():
        return load_from_disk(str(DATASET_CACHE_DIR))

    files = sorted(QA_JSON_DIR.glob("qa_pair_*.json"))
    rows: list[dict] = []
    for file in files:
        obj = json.loads(file.read_text(encoding="utf-8"))
        q = str(obj.get("question", "")).strip()
        a = str(obj.get("answer", "")).strip()
        if not q or not a:
            continue
        rendered = format_chat(q, a)
        encoded = tokenizer(
            rendered,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_attention_mask=True,
        )
        rows.append(
            {
                "source_file": str(file),
                "text": rendered,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"],
            }
        )

    if not rows:
        raise ValueError(f"No valid qa_pair_*.json found in {QA_JSON_DIR}")

    ds = Dataset.from_list(rows)
    DATASET_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATASET_CACHE_DIR))
    return ds


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_or_load_dataset(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if USE_LORA:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        report_to=["tensorboard"],
        logging_dir=str(OUTPUT_DIR / "tb_logs"),
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    final_dir = OUTPUT_DIR / ("final_lora_adapter" if USE_LORA else "final_full_model")
    trainer.model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"Training completed. Output saved to: {final_dir}")


if __name__ == "__main__":
    main()
