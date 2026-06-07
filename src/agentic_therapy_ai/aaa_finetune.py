from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


@dataclass
class AAAFineTuneConfig:
    model_path: str
    md_dir: Path
    aaa_dir: Path
    dataset_disk_path: Path = Path("outputs/aaa_finetune_dataset")
    output_dir: Path = Path("outputs/aaa_finetune_runs")
    max_length: int = 512
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 30
    save_every_epochs: int = 5
    save_total_limit: int = 2
    logging_steps: int = 10
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def _iter_files(root: Path, suffix: str) -> list[Path]:
    return sorted(root.rglob(f"*{suffix}"))


def _chunk_ids(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids), size) if ids[i : i + size]]


def build_or_load_dataset(cfg: AAAFineTuneConfig) -> Dataset:
    if cfg.dataset_disk_path.exists():
        return load_from_disk(str(cfg.dataset_disk_path))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples: list[list[int]] = []

    md_files = _iter_files(cfg.md_dir, ".md")
    md_text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in md_files)
    md_ids = tokenizer(md_text, add_special_tokens=False)["input_ids"]
    for chunk in _chunk_ids(md_ids, cfg.max_length):
        if len(chunk) < cfg.max_length:
            chunk = chunk + [tokenizer.pad_token_id] * (cfg.max_length - len(chunk))
        samples.append(chunk)

    aaa_files = _iter_files(cfg.aaa_dir, ".aaa")
    for path in aaa_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=cfg.max_length)["input_ids"]
        if len(ids) < cfg.max_length:
            ids = ids + [tokenizer.pad_token_id] * (cfg.max_length - len(ids))
        samples.append(ids)

    ds = Dataset.from_dict({"input_ids": samples})
    ds = ds.map(lambda x: {"attention_mask": [1 if t != tokenizer.pad_token_id else 0 for t in x["input_ids"]]}, num_proc=1)
    ds = ds.map(lambda x: {"labels": x["input_ids"]}, num_proc=1)
    cfg.dataset_disk_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cfg.dataset_disk_path))
    meta = {
        "num_samples": len(ds),
        "num_md_files": len(md_files),
        "num_aaa_files": len(aaa_files),
        "config": asdict(cfg),
    }
    (cfg.dataset_disk_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ds


def train_aaa_model(cfg: AAAFineTuneConfig) -> None:
    dataset = build_or_load_dataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)

    steps_per_epoch = max(1, len(dataset) // (cfg.per_device_train_batch_size * max(1, cfg.gradient_accumulation_steps)))
    save_steps = max(1, steps_per_epoch * cfg.save_every_epochs)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=True,
        report_to=["tensorboard"],
        logging_dir=str(cfg.output_dir / "tb_logs"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final_model"))
