from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from agentic_therapy_ai.drc_finetune import DRCFineTuneConfig, build_or_load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP training with accelerate.save_state checkpoints.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--md-dir", type=Path, required=True)
    parser.add_argument("--drc-dir", type=Path, required=True)
    parser.add_argument("--dataset-disk-path", type=Path, default=Path("outputs/drc_finetune_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drc_finetune_runs"))
    parser.add_argument("--tune-mode", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    cfg = DRCFineTuneConfig(
        model_path=args.model_path,
        md_dir=args.md_dir,
        drc_dir=args.drc_dir,
        dataset_disk_path=args.dataset_disk_path,
        output_dir=args.output_dir,
        use_lora=args.tune_mode == "lora",
    )

    accelerator = Accelerator(log_with="tensorboard", project_dir=str(cfg.output_dir / "tb_logs"))
    accelerator.init_trackers("drc_qwen3_finetune")

    dataset = build_or_load_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    if cfg.use_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
            ),
        )
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(dataloader, start=1):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            accelerator.log({"train/loss": loss.item(), "epoch": epoch}, step=(epoch - 1) * len(dataloader) + step)

        if epoch % args.save_every_epochs == 0:
            ckpt_dir = cfg.output_dir / f"epoch_{epoch}"
            accelerator.save_state(str(ckpt_dir))
            checkpoints = sorted([p for p in cfg.output_dir.glob("epoch_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
            while len(checkpoints) > args.keep_last:
                old = checkpoints.pop(0)
                for child in old.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(old.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                old.rmdir()

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(cfg.output_dir / "final_model", safe_serialization=True)
    tokenizer.save_pretrained(cfg.output_dir / "final_model")


if __name__ == "__main__":
    main()
