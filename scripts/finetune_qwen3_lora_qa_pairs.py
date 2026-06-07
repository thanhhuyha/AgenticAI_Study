from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Hardcoded configuration (edit before run)
MODEL_PATH = "/path/to/local/Qwen3-30B"
DATASET_DIR = Path("outputs/instruction_pairs_dataset")
OUTPUT_DIR = Path("outputs/qwen3_lora_qa_ft")

# Small-data hyperparameters (about 200 QA pairs)
MAX_STEPS = 1200
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.15


def main() -> None:
    if not DATASET_DIR.exists():
        raise ValueError(f"Dataset not found: {DATASET_DIR}. Run preprocess script first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(DATASET_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

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

    final_dir = OUTPUT_DIR / "final_lora_adapter"
    trainer.model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"Training completed. LoRA adapter saved at: {final_dir}")


if __name__ == "__main__":
    main()
