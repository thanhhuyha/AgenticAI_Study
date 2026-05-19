from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOConfig, ORPOTrainer

# ==========================================================
# Hardcoded standalone configuration
# ==========================================================
MODEL_PATH = "/models/Qwen3-30B"
TOKENIZER_PATH = "/models/Qwen3-30B-tokenizer"

# Folder A: preferred (chosen) responses
PREFERRED_JSON_DIR = "data/orpo_preferred"
# Folder B: rejected responses
REJECTED_JSON_DIR = "data/orpo_rejected"

DATASET_CACHE_DIR = "outputs/orpo_dataset_preferred_vs_rejected"
OUTPUT_DIR = "outputs/qwen30b_orpo_full_fsdp"

MAX_PROMPT_LENGTH = 1024
MAX_LENGTH = 2048

# Hyperparameters tuned for ~5000 preference pairs and full-parameter finetuning on 4 GPUs.
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 8e-7
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.08
BETA = 0.1

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
SEED = 42

# FSDP to run on 4 GPUs configured externally with accelerate config.
FSDP_MODE = "full_shard auto_wrap"
FSDP_LAYER_CLS_TO_WRAP = "Qwen3DecoderLayer,Qwen2DecoderLayer"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_response(answer: str, reasoning: str) -> str:
    if reasoning:
        return f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}".strip()
    return f"Answer:\n{answer}".strip()


def build_or_load_dataset() -> Dataset:
    cache_dir = Path(DATASET_CACHE_DIR)
    if cache_dir.exists():
        print(f"Loading cached ORPO dataset: {cache_dir}")
        return load_from_disk(str(cache_dir))

    preferred_dir = Path(PREFERRED_JSON_DIR)
    rejected_dir = Path(REJECTED_JSON_DIR)

    if not preferred_dir.exists() or not preferred_dir.is_dir():
        raise FileNotFoundError(f"Preferred folder not found: {preferred_dir}")
    if not rejected_dir.exists() or not rejected_dir.is_dir():
        raise FileNotFoundError(f"Rejected folder not found: {rejected_dir}")

    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    files = sorted(preferred_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No json files found in preferred folder: {preferred_dir}")

    for preferred_file in files:
        rejected_file = rejected_dir / preferred_file.name
        if not rejected_file.exists():
            continue

        p_obj = _load_json(preferred_file)
        r_obj = _load_json(rejected_file)

        p_question = _clean_text(p_obj.get("question", ""))
        r_question = _clean_text(r_obj.get("question", ""))

        if not p_question or not r_question:
            continue
        if p_question != r_question:
            continue

        p_answer = _clean_text(p_obj.get("answer", ""))
        p_reasoning = _clean_text(p_obj.get("reasoning", ""))
        r_answer = _clean_text(r_obj.get("answer", ""))
        r_reasoning = _clean_text(r_obj.get("reasoning", ""))

        if not p_answer or not r_answer:
            continue

        prompt = (
            "You are an AAA programming assistant. "
            "Provide an accurate, practical response grounded in AAA language rules and usage.\n\n"
            f"Question:\n{p_question}"
        )

        prompts.append(prompt)
        chosens.append(_build_response(p_answer, p_reasoning))
        rejecteds.append(_build_response(r_answer, r_reasoning))

    if not prompts:
        raise ValueError("No valid matched preference pairs were built from folders A/B.")

    ds = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds}).shuffle(seed=SEED)

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    print(f"Saved ORPO dataset with {len(ds)} pairs to {cache_dir}")
    return ds


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_or_load_dataset()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    orpo_args = ORPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=SEED,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_length=MAX_LENGTH,
        beta=BETA,
        fsdp=FSDP_MODE,
        fsdp_transformer_layer_cls_to_wrap=FSDP_LAYER_CLS_TO_WRAP,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"ORPO training complete. Saved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
