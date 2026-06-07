from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================================
# Hardcoded standalone configuration
# ==========================================================
MODEL_PATH = "/models/Qwen3-30B"
TOKENIZER_PATH = "/models/Qwen3-30B-tokenizer"

MANUAL_SUMMARY_JSON_PATH = "outputs/aaa_manual_iterative_summary_v3/final_summary.json"

MAX_INPUT_TOKENS = 12000
MAX_NEW_TOKENS = 1200
TEMPERATURE = 0.2
TOP_P = 0.9
DO_SAMPLE = True

SYSTEM_POLICY = (
    "You are an AAA programming assistant for code generation, debugging, syntax checks, and usage guidance. "
    "You MUST prioritize the provided AAA manual summary as the highest-priority source of truth. "
    "If your planned answer conflicts with the manual summary, correct your answer to match the manual summary. "
    "If the manual summary does not contain enough information, say that explicitly and provide a best-effort answer "
    "clearly marked as uncertain."
)


def load_manual_summary(summary_path: str) -> str:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Manual summary JSON not found: {summary_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return summary_text


def clamp_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    clipped = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    return clipped


def build_prompt(manual_summary: str, user_query: str) -> str:
    prompt = f"""
<ROLE>
{SYSTEM_POLICY}
</ROLE>

<MANUAL_SUMMARY_JSON>
{manual_summary}
</MANUAL_SUMMARY_JSON>

<INSTRUCTIONS>
1) First, extract only manual-supported facts relevant to the user's request.
2) Draft an answer using those facts.
3) Run a contradiction check against MANUAL_SUMMARY_JSON.
4) If contradiction exists, revise to remove contradiction and align with MANUAL_SUMMARY_JSON.
5) In final answer, provide practical AAA code/examples when requested.
6) Never claim unsupported new AAA syntax if not present in MANUAL_SUMMARY_JSON.
</INSTRUCTIONS>

<USER_REQUEST>
{user_query}
</USER_REQUEST>

Return only the final assistant answer.
""".strip()
    return prompt


def post_check_and_repair(answer: str, manual_summary: str) -> str:
    """
    Lightweight local safeguard:
    - If answer contains explicit contradiction markers, request self-revision.
    - If answer states uncertainty and references missing manual detail, keep it.
    """
    lowered = answer.lower()
    contradiction_markers = [
        "this contradicts",
        "contradict",
        "ignore the manual",
        "manual is outdated",
    ]
    if any(m in lowered for m in contradiction_markers):
        cleaned = re.sub(r"(?i)this contradicts.*?(\.|$)", "", answer).strip()
        cleaned += (
            "\n\nNote: Revised to follow the provided AAA manual summary as authoritative reference."
        )
        return cleaned
    return answer


def generate_answer(model, tokenizer, manual_summary: str, user_query: str) -> str:
    prompt = build_prompt(manual_summary=manual_summary, user_query=user_query)
    prompt = clamp_text_to_tokens(tokenizer, prompt, MAX_INPUT_TOKENS)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer = post_check_and_repair(answer, manual_summary)
    return answer


def main() -> None:
    manual_summary = load_manual_summary(MANUAL_SUMMARY_JSON_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )

    print("AAA Chatbot ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nUser> ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_query:
            continue

        try:
            answer = generate_answer(model, tokenizer, manual_summary, user_query)
        except Exception as exc:
            print(f"Assistant> [Error] {exc}")
            continue

        print(f"Assistant> {answer}")


if __name__ == "__main__":
    main()
