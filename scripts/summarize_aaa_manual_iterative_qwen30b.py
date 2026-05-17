from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ==========================================================
# Hardcoded configuration (standalone, no CLI required)
# ==========================================================
MODEL_PATH = "/models/Qwen3-30B"
TOKENIZER_PATH = "/models/Qwen3-30B-tokenizer"

MANUAL_MD_DIR = "data/aaa_manual_md"
OUTPUT_DIR = "outputs/aaa_manual_iterative_summary_v3"

# Qwen3 30B context window estimate (override if your local variant differs)
MODEL_CONTEXT_WINDOW_TOKENS = 32768
TARGET_SUMMARY_RATIO = 0.50
TARGET_SUMMARY_TOKENS = int(MODEL_CONTEXT_WINDOW_TOKENS * TARGET_SUMMARY_RATIO)

# Generation controls
MAX_NEW_TOKENS = 1800
TEMPERATURE = 0.2
TOP_P = 0.9
DO_SAMPLE = True


@dataclass
class IterationResult:
    index: int
    file_name: str
    file_tokens: int
    prev_summary_tokens: int
    new_summary_tokens: int
    target_tokens: int
    within_target: bool
    summary_json: dict[str, Any]
    summary_markdown: str


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _count_tokens(tokenizer, text: str) -> int:
    if not text.strip():
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _extract_json_block(text: str) -> dict[str, Any]:
    # try direct parse
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # fallback fenced-json parse
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # fallback first object-like span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # last-resort safe structure
    return {
        "aaa_syntax_rules": [],
        "aaa_usage_patterns": [],
        "aaa_debugging_guidance": [],
        "aaa_common_pitfalls": [],
        "aaa_code_templates": [],
        "aaa_key_constraints": [],
        "aaa_term_glossary": [],
        "aaa_summary_notes": [text[:4000]],
    }


def _json_to_markdown(summary_json: dict[str, Any]) -> str:
    section_titles = {
        "aaa_syntax_rules": "AAA Syntax Rules",
        "aaa_usage_patterns": "AAA Usage Patterns",
        "aaa_debugging_guidance": "AAA Debugging Guidance",
        "aaa_common_pitfalls": "AAA Common Pitfalls",
        "aaa_code_templates": "AAA Code Templates",
        "aaa_key_constraints": "AAA Key Constraints",
        "aaa_term_glossary": "AAA Term Glossary",
        "aaa_summary_notes": "AAA Summary Notes",
    }

    lines = ["# AAA Manual Iterative Summary", ""]
    for key in section_titles:
        lines.append(f"## {section_titles[key]}")
        values = summary_json.get(key, [])
        if isinstance(values, list) and values:
            for v in values:
                lines.append(f"- {str(v).strip()}")
        elif isinstance(values, str) and values.strip():
            lines.append(f"- {values.strip()}")
        else:
            lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_prompt(prev_summary_json: dict[str, Any], current_md: str, target_tokens: int) -> str:
    prev_summary_text = json.dumps(prev_summary_json, ensure_ascii=False, indent=2)
    return f"""
You are an expert AAA language summarizer for an AAA AI assistant that performs AAA code generation,
AAA code debugging, and AAA syntax/usage help.

Task:
1) Read CURRENT_PAGE markdown content.
2) Read PREVIOUS_SUMMARY_JSON which summarizes all previous pages.
3) Produce NEXT_SUMMARY_JSON that fuses old+new knowledge.
4) Keep only high-value technical details needed for writing and debugging AAA.
5) Remove repetition and low-value prose.
6) Keep NEXT_SUMMARY concise so total token length remains <= {target_tokens} tokens.

Output requirements:
- Return STRICT JSON only.
- Use exactly this schema:
{{
  "aaa_syntax_rules": ["..."],
  "aaa_usage_patterns": ["..."],
  "aaa_debugging_guidance": ["..."],
  "aaa_common_pitfalls": ["..."],
  "aaa_code_templates": ["..."],
  "aaa_key_constraints": ["..."],
  "aaa_term_glossary": ["..."],
  "aaa_summary_notes": ["..."]
}}

PREVIOUS_SUMMARY_JSON:
{prev_summary_text}

CURRENT_PAGE:
{current_md}
""".strip()


def _compress_if_needed(tokenizer, generator, summary_json: dict[str, Any], target_tokens: int) -> dict[str, Any]:
    current_text = json.dumps(summary_json, ensure_ascii=False)
    current_tokens = _count_tokens(tokenizer, current_text)
    if current_tokens <= target_tokens:
        return summary_json

    compress_prompt = f"""
Compress the AAA summary JSON below while preserving the most important technical knowledge for AAA generation/debugging.
Return STRICT JSON with the same schema keys.
Hard token budget: <= {target_tokens} tokens.

JSON:
{json.dumps(summary_json, ensure_ascii=False, indent=2)}
""".strip()

    out = generator(compress_prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, temperature=TEMPERATURE, top_p=TOP_P)
    text = out[0]["generated_text"][len(compress_prompt) :].strip()
    return _extract_json_block(text)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir = out_dir / "iterations"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    md_files = sorted(Path(MANUAL_MD_DIR).glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {MANUAL_MD_DIR}")

    # Initial empty summary
    prev_summary_json: dict[str, Any] = {
        "aaa_syntax_rules": [],
        "aaa_usage_patterns": [],
        "aaa_debugging_guidance": [],
        "aaa_common_pitfalls": [],
        "aaa_code_templates": [],
        "aaa_key_constraints": [],
        "aaa_term_glossary": [],
        "aaa_summary_notes": [],
    }

    all_iterations: list[IterationResult] = []

    for i, md_path in enumerate(md_files):
        page_text = _clean_text(md_path.read_text(encoding="utf-8", errors="ignore"))

        # Reserve prompt room by clipping overly large page chunk if needed.
        prompt_budget_for_page = max(2048, MODEL_CONTEXT_WINDOW_TOKENS - TARGET_SUMMARY_TOKENS - 4096)
        page_text = _truncate_to_tokens(tokenizer, page_text, prompt_budget_for_page)

        prev_summary_tokens = _count_tokens(tokenizer, json.dumps(prev_summary_json, ensure_ascii=False))
        file_tokens = _count_tokens(tokenizer, page_text)

        prompt = _build_prompt(prev_summary_json, page_text, TARGET_SUMMARY_TOKENS)
        gen = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, temperature=TEMPERATURE, top_p=TOP_P)
        model_text = gen[0]["generated_text"][len(prompt) :].strip()

        next_summary_json = _extract_json_block(model_text)
        next_summary_json = _compress_if_needed(tokenizer, generator, next_summary_json, TARGET_SUMMARY_TOKENS)

        next_summary_md = _json_to_markdown(next_summary_json)
        new_summary_tokens = _count_tokens(tokenizer, json.dumps(next_summary_json, ensure_ascii=False))

        result = IterationResult(
            index=i,
            file_name=md_path.name,
            file_tokens=file_tokens,
            prev_summary_tokens=prev_summary_tokens,
            new_summary_tokens=new_summary_tokens,
            target_tokens=TARGET_SUMMARY_TOKENS,
            within_target=new_summary_tokens <= TARGET_SUMMARY_TOKENS,
            summary_json=next_summary_json,
            summary_markdown=next_summary_md,
        )
        all_iterations.append(result)

        iter_json_path = iteration_dir / f"iter_{i:04d}_{md_path.stem}.json"
        iter_md_path = iteration_dir / f"iter_{i:04d}_{md_path.stem}.md"
        iter_meta_path = iteration_dir / f"iter_{i:04d}_{md_path.stem}_meta.json"

        iter_json_path.write_text(json.dumps(next_summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
        iter_md_path.write_text(next_summary_md, encoding="utf-8")
        iter_meta_path.write_text(
            json.dumps(
                {
                    "iteration": i,
                    "file": md_path.name,
                    "file_tokens": file_tokens,
                    "previous_summary_tokens": prev_summary_tokens,
                    "new_summary_tokens": new_summary_tokens,
                    "target_tokens": TARGET_SUMMARY_TOKENS,
                    "within_target": new_summary_tokens <= TARGET_SUMMARY_TOKENS,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        prev_summary_json = next_summary_json

    final_json_path = out_dir / "final_summary.json"
    final_md_path = out_dir / "final_summary.md"
    trace_path = out_dir / "iteration_trace.json"

    final_json_path.write_text(json.dumps(prev_summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    final_md_path.write_text(_json_to_markdown(prev_summary_json), encoding="utf-8")

    trace_payload = {
        "model_path": MODEL_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "context_window_tokens": MODEL_CONTEXT_WINDOW_TOKENS,
        "target_ratio": TARGET_SUMMARY_RATIO,
        "target_tokens": TARGET_SUMMARY_TOKENS,
        "total_files": len(md_files),
        "final_summary_tokens": _count_tokens(tokenizer, json.dumps(prev_summary_json, ensure_ascii=False)),
        "final_within_target": _count_tokens(tokenizer, json.dumps(prev_summary_json, ensure_ascii=False))
        <= TARGET_SUMMARY_TOKENS,
        "iterations": [
            {
                "iteration": r.index,
                "file_name": r.file_name,
                "file_tokens": r.file_tokens,
                "prev_summary_tokens": r.prev_summary_tokens,
                "new_summary_tokens": r.new_summary_tokens,
                "target_tokens": r.target_tokens,
                "within_target": r.within_target,
            }
            for r in all_iterations
        ],
    }
    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Final summary saved to: {final_json_path} and {final_md_path}")
    print(f"Iteration outputs saved to: {iteration_dir}")


if __name__ == "__main__":
    main()
