from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUMMARY_SYSTEM_PROMPT = """You are an expert DRC engineer. Extract only high-value information from manual pages.
Return concise, implementation-ready notes for writing and debugging DRC rules.
Focus on syntax, semantics, rule parameters, edge cases, precedence, defaults, and examples.
Skip marketing and repeated prose."""

CHUNK_TEMPLATE = """Summarize this DRC manual page.

Page path: {page_path}

Content:
{page_text}

Output JSON with keys:
- key_rules: list[str]
- syntax_patterns: list[str]
- constraints_and_limits: list[str]
- debugging_signals: list[str]
- examples: list[str]
- unresolved_questions: list[str]
"""

FINAL_TEMPLATE = """You are given extracted notes from many DRC manual pages.
Merge and deduplicate into a compact reference for LLM prompting.
Target <= {target_chars} characters.

Notes JSON:
{notes_json}

Output JSON with keys:
- drc_cheat_sheet: string
- prompt_context_for_code_gen: string
- prompt_context_for_debugging: string
- dropped_or_ambiguous_items: list[str]
"""


@dataclass
class DRCCompactionConfig:
    model_path: str
    md_dir: Path
    output_path: Path = Path("outputs/drc_compacted_context.json")
    max_input_chars_per_page: int = 6000
    max_new_tokens_chunk: int = 500
    max_new_tokens_final: int = 1200
    temperature: float = 0.1
    top_p: float = 0.9
    target_chars: int = 14000
    max_pages: int | None = None


class LocalQwenCompactor:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            )
        completion = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def iter_markdown_files(md_dir: Path) -> Iterable[Path]:
    yield from sorted(md_dir.rglob("*.md"))


def compact_manual(config: DRCCompactionConfig) -> dict:
    compactor = LocalQwenCompactor(model_path=config.model_path)

    pages = list(iter_markdown_files(config.md_dir))
    if config.max_pages is not None:
        pages = pages[: config.max_pages]

    if not pages:
        raise ValueError(f"No markdown files found in: {config.md_dir}")

    extracted_notes: list[dict] = []
    for page in pages:
        raw = page.read_text(encoding="utf-8", errors="ignore")
        clipped = raw[: config.max_input_chars_per_page]
        user_prompt = CHUNK_TEMPLATE.format(page_path=str(page), page_text=clipped)
        response = compactor._generate(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=config.max_new_tokens_chunk,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        extracted_notes.append({"page": str(page), "summary": response})

    final_user_prompt = FINAL_TEMPLATE.format(
        target_chars=config.target_chars,
        notes_json=json.dumps(extracted_notes, ensure_ascii=False),
    )

    consolidated = compactor._generate(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=final_user_prompt,
        max_new_tokens=config.max_new_tokens_final,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    result = {
        "model_path": config.model_path,
        "pages_processed": len(extracted_notes),
        "target_chars": config.target_chars,
        "final_context": consolidated,
        "raw_page_notes": extracted_notes,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
