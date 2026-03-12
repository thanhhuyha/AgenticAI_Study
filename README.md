# Agentic Therapy AI (Local, 4x T4)

Production-oriented Python codebase for an **agentic multimodal pipeline**:
1. Accepts a local indoor image of a person performing a therapy exercise.
2. Uses a local **state-of-the-art VLM** (Hugging Face) to produce a strict structured description JSON (pose, image quality, lighting, clothes, camera angle, clutter/background, etc.).
3. Uses a local **state-of-the-art LLM** (vLLM) to transform that JSON into a Veo3 prompt bundle for synthetic frame generation with controlled variations.

## Recommended Models for 4x T4

- VLM: `Qwen/Qwen2-VL-72B-Instruct`
- LLM: `meta-llama/Meta-Llama-3.1-70B-Instruct`

> These are configurable via environment variables. Quantization and serving strategy can be adjusted based on memory constraints.

## Architecture

- `VLMDescriptor`: image -> `TherapyFrameDescription` JSON.
- `Veo3PromptGenerator`: JSON -> `Veo3PromptBundle` JSON.
- `TherapyAgentPipeline`: orchestration and output persistence.
- CLI entrypoint: `therapy-agent run <image_path>`.

## JSON Contracts

Two validated schema outputs (Pydantic):
- `TherapyFrameDescription` (fixed schema, versioned)
- `Veo3PromptBundle`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Run

```bash
therapy-agent run ./data/sample_frame.jpg --run-id demo01
```

Outputs:
- `outputs/<run_id>/frame_description.json`
- `outputs/<run_id>/veo3_prompt_bundle.json`

## Operational Notes

- Designed for local model execution with multi-GPU support.
- vLLM tensor parallel size defaults to `4` for 4x T4.
- Add model auth tokens in your environment as required by Hugging Face.

## Quality Controls

- Strict schema validation on model outputs.
- Retry logic around VLM generation.
- Deterministic VLM decoding by default.

## Next Production Steps

- Add API server (FastAPI) + job queue.
- Add observability (OpenTelemetry + structured logs).
- Add post-processors (pose estimator cross-check, prompt safety filters).
- Add dataset-based regression tests for therapy pose fidelity.
