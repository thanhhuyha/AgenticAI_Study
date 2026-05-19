from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================================
# Hardcoded standalone configuration
# ==========================================================
MODEL_PATH = "/models/Qwen3-30B"
TOKENIZER_PATH = "/models/Qwen3-30B-tokenizer"

INPUT_FOLDER = "data/json_qa_input"
MAX_NEW_TOKENS = 1000
TEMPERATURE = 0.2
TOP_P = 0.9
DO_SAMPLE = True


def prefixed_sibling_folder(input_path: Path, prefix: str) -> Path:
    return input_path.parent / f"{prefix}{input_path.name}"


def load_question_from_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")

    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing non-empty 'question' in {path}")
    return question.strip()


def save_question_only(path: Path, question: str) -> None:
    path.write_text(json.dumps({"question": question}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(question: str) -> str:
    # NOTE: We explicitly avoid requesting private chain-of-thought.
    return f"""
You are a helpful AAA programming assistant.
Answer the user question clearly and correctly.

Output STRICT JSON with exactly these keys:
- "answer": final direct answer.
- "reasoning": concise, high-level rationale and grounding summary (NO private chain-of-thought).

Question:
{question}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {"answer": text[:4000], "reasoning": "Unable to parse strict JSON; raw answer captured."}


def generate_answer(model, tokenizer, question: str) -> dict[str, str]:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    parsed = extract_json(raw)

    answer = str(parsed.get("answer", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()

    if not answer:
        answer = raw[:4000]
    if not reasoning:
        reasoning = "High-level rationale unavailable; answer generated from model inference."

    return {"answer": answer, "reasoning": reasoning}


def main() -> None:
    input_dir = Path(INPUT_FOLDER)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    question_out_dir = prefixed_sibling_folder(input_dir, "Q")
    qa_out_dir = prefixed_sibling_folder(input_dir, "QA_")
    question_out_dir.mkdir(parents=True, exist_ok=True)
    qa_out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files in {input_dir}")

    for src_file in files:
        question = load_question_from_json(src_file)

        # Save question-only JSON in Q<folder>
        q_file = question_out_dir / src_file.name
        save_question_only(q_file, question)

        # Generate answer + concise rationale/grounding summary
        generated = generate_answer(model, tokenizer, question)
        qa_payload = {
            "question": question,
            "answer": generated["answer"],
            "reasoning": generated["reasoning"],
        }

        qa_file = qa_out_dir / src_file.name
        qa_file.write_text(json.dumps(qa_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Processed: {src_file.name}")

    print(f"Done. Question files: {question_out_dir}")
    print(f"Done. QA files: {qa_out_dir}")


if __name__ == "__main__":
    main()
