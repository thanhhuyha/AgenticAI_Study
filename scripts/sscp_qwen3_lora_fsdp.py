from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_cosine_schedule_with_warmup

# =========================
# Hardcoded configuration
# =========================
MODEL_PATH = "/path/to/local/Qwen3-30B"
MD_ROOT_DIR = Path("/path/to/manual_md")
DATASET_CACHE_DIR = Path("outputs/sscp_dataset_1024")
RUN_OUTPUT_DIR = Path("outputs/sscp_qwen3_30b_lora_fsdp")

MAX_LENGTH = 1024
TARGET_TOTAL_TOKENS = 1_000_000

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.08
MAX_GRAD_NORM = 1.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.10

CHECKPOINT_EVERY_STEPS = 200
KEEP_LAST_N_CHECKPOINTS = 2
LOG_EVERY_STEPS = 10
SEED = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _iter_md_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _chunk_ids(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids), size) if ids[i : i + size]]


def build_or_load_dataset(tokenizer: AutoTokenizer) -> Dataset:
    if DATASET_CACHE_DIR.exists():
        return load_from_disk(str(DATASET_CACHE_DIR))

    md_files = _iter_md_files(MD_ROOT_DIR)
    if not md_files:
        raise ValueError(f"No markdown files found under {MD_ROOT_DIR}")

    merged_text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in md_files)
    token_ids = tokenizer(merged_text, add_special_tokens=False)["input_ids"]

    token_ids = token_ids[:TARGET_TOTAL_TOKENS]
    chunks = _chunk_ids(token_ids, MAX_LENGTH)

    if not chunks:
        raise ValueError("No training chunks generated. Check dataset/model tokenizer.")

    pad_id = tokenizer.pad_token_id
    padded_chunks = []
    for c in chunks:
        if len(c) < MAX_LENGTH:
            c = c + [pad_id] * (MAX_LENGTH - len(c))
        padded_chunks.append(c)

    ds = Dataset.from_dict({"input_ids": padded_chunks})
    ds = ds.map(
        lambda x: {
            "attention_mask": [1 if t != pad_id else 0 for t in x["input_ids"]],
            "labels": x["input_ids"],
        },
        num_proc=1,
    )
    DATASET_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATASET_CACHE_DIR))
    return ds


def prune_old_checkpoints() -> None:
    ckpts = sorted([p for p in RUN_OUTPUT_DIR.glob("step_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    while len(ckpts) > KEEP_LAST_N_CHECKPOINTS:
        old = ckpts.pop(0)
        for f in old.rglob("*"):
            if f.is_file():
                f.unlink()
        for d in sorted(old.rglob("*"), reverse=True):
            if d.is_dir():
                d.rmdir()
        old.rmdir()


def main() -> None:
    set_seed(SEED)
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS, log_with="tensorboard", project_dir=str(RUN_OUTPUT_DIR / "tb_logs"))
    accelerator.init_trackers("qwen3_30b_sscp_lora")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_or_load_dataset(tokenizer)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=PER_DEVICE_BATCH_SIZE, collate_fn=default_data_collator)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_cfg)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_update_steps = math.ceil((len(train_loader) * NUM_EPOCHS) / GRAD_ACCUM_STEPS)
    warmup_steps = max(1, int(total_update_steps * WARMUP_RATIO))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % LOG_EVERY_STEPS == 0:
                    accelerator.log({"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                if global_step % CHECKPOINT_EVERY_STEPS == 0:
                    ckpt_dir = RUN_OUTPUT_DIR / f"step_{global_step}"
                    accelerator.save_state(str(ckpt_dir))
                    prune_old_checkpoints()

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    final_dir = RUN_OUTPUT_DIR / "final_lora_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
