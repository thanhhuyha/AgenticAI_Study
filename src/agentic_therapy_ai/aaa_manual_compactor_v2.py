from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT_V2 = """You are a principal AAA engineer and technical summarizer.
Your output will be consumed by another Qwen3-30B model for AAA code writing/debugging.
Preserve only high-signal knowledge: syntax, APIs, rule semantics, constraints, precedence, defaults, examples, pitfalls, and debugging cues.
Avoid repetition, prose, and irrelevant content.
"""

ITER_TEMPLATE = """You are processing a AAA manual incrementally.

Existing cumulative summary (covers all previous pages):
{previous_summary}

New markdown chapter/page-group to absorb:
Path: {page_path}
Content:
{page_text}

Task:
Fuse the new content into the cumulative summary and return an updated summary.
Requirements:
- Keep all critical AAA knowledge needed for code writing/debugging.
- Deduplicate aggressively.
- Keep concise and structured.
- If previous summary has errors/ambiguities, correct them.

Return JSON:
{{
  "updated_summary": "string",
  "newly_added_critical_items": ["string"],
  "dropped_or_merged_items": ["string"]
}}
"""


@dataclass
class AAACompactionConfigV2:
    model_path: str
    md_dir: Path
    output_path: Path = Path("outputs/aaa_compacted_context_v2.json")
    output_markdown_path: Path = Path("outputs/aaa_compacted_context_v2.md")
    max_input_chars_per_file: int = 12000
    max_new_tokens_per_iter: int = 1400
    temperature: float = 0.1
    top_p: float = 0.9
    model_context_window_tokens: int = 32768
    target_context_ratio: float = 0.50
    max_files: int | None = None


class LocalQwenCompactorV2:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, system_prompt: str, user_prompt: str, cfg: AAACompactionConfigV2) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens_per_iter,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        completion = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def iter_markdown_files(md_dir: Path) -> Iterable[Path]:
    yield from sorted(md_dir.rglob("*.md"))


def _extract_updated_summary(raw_response: str) -> str:
    # Best effort JSON extraction; fallback to raw response.
    try:
        payload = json.loads(raw_response)
        summary = payload.get("updated_summary", "")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except json.JSONDecodeError:
        pass
    return raw_response.strip()


def compact_manual_v2(config: AAACompactionConfigV2) -> dict:
    compactor = LocalQwenCompactorV2(model_path=config.model_path)
    files = list(iter_markdown_files(config.md_dir))
    if config.max_files is not None:
        files = files[: config.max_files]
    if not files:
        raise ValueError(f"No markdown files found in: {config.md_dir}")

    target_tokens = max(256, int(config.model_context_window_tokens * config.target_context_ratio))

    cumulative_summary = "No previous pages yet."
    trace: list[dict] = []
    for idx, file_path in enumerate(files):
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        clipped = raw[: config.max_input_chars_per_file]
        prompt = ITER_TEMPLATE.format(
            previous_summary=cumulative_summary,
            page_path=str(file_path),
            page_text=clipped,
        )
        raw_response = compactor.generate(SYSTEM_PROMPT_V2, prompt, config)
        updated_summary = _extract_updated_summary(raw_response)
        cumulative_summary = updated_summary
        trace.append(
            {
                "iteration": idx,
                "file": str(file_path),
                "updated_summary_tokens": compactor.count_tokens(updated_summary),
                "raw_response": raw_response,
            }
        )

    final_tokens = compactor.count_tokens(cumulative_summary)
    result = {
        "model_path": config.model_path,
        "files_processed": len(files),
        "model_context_window_tokens": config.model_context_window_tokens,
        "target_context_ratio": config.target_context_ratio,
        "target_tokens": target_tokens,
        "final_summary_tokens": final_tokens,
        "within_target_50_percent": final_tokens <= target_tokens,
        "final_summary": cumulative_summary,
        "iteration_trace": trace,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    config.output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_markdown_path.write_text(
        (
            "# AAA Manual Iterative Compaction (V2)\n\n"
            f"- Files processed: **{result['files_processed']}**\n"
            f"- Target tokens (50% window): **{target_tokens}**\n"
            f"- Final summary tokens: **{final_tokens}**\n"
            f"- Within target: **{result['within_target_50_percent']}**\n\n"
            "## Final Summary\n\n"
            f"{cumulative_summary}\n"
        ),
        encoding="utf-8",
    )
    return result
