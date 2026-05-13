from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hardcoded config
MODEL_PATH = "/path/to/local/Qwen3-30B"
MD_DIR = Path("/path/to/aaa_manual_md")
OUTPUT_DIR = Path("outputs/generated_instruction_qa")
N_TOTAL_PAIRS = 200
PAIRS_PER_INFERENCE = 5
MAX_INPUT_CHARS_PER_BATCH = 30000
MAX_NEW_TOKENS = 1800
TEMPERATURE = 0.7
TOP_P = 0.9


SYSTEM_PROMPT = """You are a AAA instruction-data generator.
CRITICAL RULE: Use ONLY information from the provided manual excerpts.
If a concept is not present in the excerpts, do not invent it.
Generate realistic AAA-focused instruction Q/A pairs for fine-tuning.
Question categories should include:
1) Debug a AAA snippet
2) Write a AAA snippet for a design-rule objective
3) Syntax checking/correction
4) Command usage explanation
Difficulty should vary from medium to complex.
Output strict JSON only.
"""


USER_TEMPLATE = """Manual excerpts (authoritative source):
{manual_text}

Generate exactly {k} instruction pairs.
Return JSON array with items:
[
  {{
    "question_type": "debug|write_rule|syntax_check|command_usage",
    "question": "...",
    "answer": "...",
    "source_grounding": ["short quote/paraphrase from manual excerpt used"]
  }}
]
"""


def _iter_md_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _build_context(files: list[Path]) -> str:
    text_blocks = []
    used = 0
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        clipped = raw[:4000]
        block = f"\n\n# File: {fp.name}\n{clipped}"
        if used + len(block) > MAX_INPUT_CHARS_PER_BATCH:
            break
        text_blocks.append(block)
        used += len(block)
    if not text_blocks:
        raise ValueError(f"No markdown content loaded from {MD_DIR}")
    return "".join(text_blocks)


def _generate_json_text(model, tokenizer, manual_context: str, k: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(manual_text=manual_context, k=k)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_json_array(text: str) -> list[dict]:
    match = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON array.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, list):
        raise ValueError("Parsed JSON is not a list.")
    return payload


def main() -> None:
    md_files = _iter_md_files(MD_DIR)
    if not md_files:
        raise ValueError(f"No .md files found in {MD_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    manual_context = _build_context(md_files)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_pairs: list[dict] = []
    round_idx = 0
    while len(generated_pairs) < N_TOTAL_PAIRS:
        round_idx += 1
        remaining = N_TOTAL_PAIRS - len(generated_pairs)
        k = min(PAIRS_PER_INFERENCE, remaining)
        raw = _generate_json_text(model, tokenizer, manual_context, k)
        try:
            pairs = _extract_json_array(raw)
        except Exception:
            continue

        for pair in pairs:
            if len(generated_pairs) >= N_TOTAL_PAIRS:
                break
            q = str(pair.get("question", "")).strip()
            a = str(pair.get("answer", "")).strip()
            if not q or not a:
                continue
            pair["id"] = f"pair_{len(generated_pairs):06d}"
            pair["generation_round"] = round_idx
            generated_pairs.append(pair)

    for idx, pair in enumerate(generated_pairs):
        out_file = OUTPUT_DIR / f"qa_pair_{idx:06d}.json"
        out_file.write_text(json.dumps(pair, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "num_pairs": len(generated_pairs),
        "output_dir": str(OUTPUT_DIR),
        "model_path": MODEL_PATH,
        "md_dir": str(MD_DIR),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
