from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

# Hardcoded config (edit paths as needed)
MODEL_PATH = "/path/to/local/Qwen3-30B"
RAW_TXT_DIR = Path("/path/to/qa_txt_files")
DATASET_OUT_DIR = Path("outputs/instruction_pairs_dataset")
MAX_LENGTH = 1024


QUESTION_RE = re.compile(r"^\s*question\b[:\-\s]*", flags=re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*answer\b[:\-\s]*", flags=re.IGNORECASE)


def _normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pair(raw_text: str) -> tuple[str, str]:
    lines = raw_text.splitlines()
    q_lines: list[str] = []
    a_lines: list[str] = []
    mode: str | None = None

    for line in lines:
        if QUESTION_RE.match(line):
            mode = "q"
            q_lines.append(QUESTION_RE.sub("", line).strip())
            continue
        if ANSWER_RE.match(line):
            mode = "a"
            a_lines.append(ANSWER_RE.sub("", line).strip())
            continue
        if mode == "q":
            q_lines.append(line.strip())
        elif mode == "a":
            a_lines.append(line.strip())

    question = _normalize_space("\n".join(q_lines))
    answer = _normalize_space("\n".join(a_lines))
    return question, answer


def _format_instruction_sample(question: str, answer: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant for AAA and EDA tasks.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{answer}\n"
        "<|im_end|>"
    )


def build_or_load_dataset() -> Dataset:
    if DATASET_OUT_DIR.exists():
        return load_from_disk(str(DATASET_OUT_DIR))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict] = []
    bad_files: list[str] = []
    txt_files = sorted(RAW_TXT_DIR.rglob("*.txt"))
    for file in txt_files:
        raw = file.read_text(encoding="utf-8", errors="ignore")
        q, a = _extract_pair(raw)
        if not q or not a:
            bad_files.append(str(file))
            continue

        rendered = _format_instruction_sample(q, a)
        encoded = tokenizer(
            rendered,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
        )
        rows.append(
            {
                "source_file": str(file),
                "question": q,
                "answer": a,
                "text": rendered,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["input_ids"],
            }
        )

    if not rows:
        raise ValueError("No valid Q/A pairs found. Check raw txt files format.")

    ds = Dataset.from_list(rows)
    DATASET_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATASET_OUT_DIR))
    metadata = {
        "num_total_txt_files": len(txt_files),
        "num_valid_samples": len(rows),
        "num_invalid_files": len(bad_files),
        "invalid_files": bad_files,
        "max_length": MAX_LENGTH,
    }
    (DATASET_OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ds


def main() -> None:
    ds = build_or_load_dataset()
    print(f"Saved/loaded dataset: {DATASET_OUT_DIR}")
    print(f"Number of samples: {len(ds)}")


if __name__ == "__main__":
    main()
