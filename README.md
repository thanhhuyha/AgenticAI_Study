# Agentic Therapy AI (Local, 4x T4)

Production-oriented Python codebase for an **agentic multimodal pipeline**:
1. Accepts a local indoor image of a person performing a therapy exercise.
2. Uses a local **state-of-the-art VLM** (Hugging Face) to produce a strict structured description JSON (pose, image quality, lighting, clothes, camera angle, clutter/background, etc.).
3. Uses a local **state-of-the-art LLM** (vLLM) to transform that JSON into a Veo3 prompt bundle for synthetic frame generation with controlled variations.

## System Overview

The system is designed as a two-stage local generation pipeline:

- **Stage A: Perception / Description (VLM)**
  - Input: one local image.
  - Process: a vision-language model receives the image plus strict schema instructions.
  - Output: validated `TherapyFrameDescription` JSON.

- **Stage B: Prompt Synthesis (LLM)**
  - Input: the validated description JSON.
  - Process: a local text LLM converts structured fields to a Veo3-oriented prompting package.
  - Output: validated `Veo3PromptBundle` JSON.

- **Stage C: Orchestration & Persistence**
  - Runs Stage A then Stage B.
  - Writes both JSON outputs to `outputs/<run_id>/`.

This architecture isolates perception from prompt engineering, making behavior easier to test, monitor, and replace model-by-model.

## Recommended Models for 4x T4

- VLM: `Qwen/Qwen2-VL-72B-Instruct`
- LLM: `meta-llama/Meta-Llama-3.1-70B-Instruct`

> These are configurable via environment variables. Quantization and serving strategy can be adjusted based on memory constraints.

## End-to-End Flow

1. `therapy-agent run <image_path> --run-id <id>` (CLI).
2. CLI loads runtime settings.
3. Pipeline loads/initializes VLM + LLM components.
4. VLM describes the image into strict JSON.
5. JSON is validated with Pydantic schemas.
6. LLM builds Veo3 prompt bundle from that JSON.
7. Prompt bundle is validated.
8. Both artifacts are written to disk.

## File-by-File Purpose

### Root

- `pyproject.toml`
  - Package metadata, dependencies, dev tooling (pytest/ruff/mypy), and CLI entrypoint (`therapy-agent`).

- `.env.example`
  - Environment variable template for model IDs and generation/runtime parameters.

- `README.md`
  - Operator/developer documentation for architecture, setup, execution, and deployment notes.

- `docker-compose.yml`
  - GPU-oriented local runner template requesting 4 NVIDIA GPUs and mounting Hugging Face cache.

### Scripts

- `scripts/run_local.sh`
  - Thin convenience wrapper around CLI execution (`therapy-agent run`).

### Package: `src/agentic_therapy_ai/`

- `__init__.py`
  - Package marker and public-export control placeholder.

- `config.py`
  - Typed runtime settings (Pydantic Settings) sourced from environment / `.env`.
  - Central place for model IDs, token limits, decoding params, output directory, and tensor parallel size.

- `schemas.py`
  - Canonical JSON contracts:
    - `TherapyFrameDescription` and nested models for pose/appearance/image/scene.
    - `Veo3PromptBundle` output structure.
  - Enforces allowed enums and value constraints.

- `prompts.py`
  - Prompt templates/instructions:
    - VLM instruction forcing fixed JSON schema output.
    - LLM system prompt for Veo3 bundle generation.

- `json_utils.py`
  - Shared utility to extract JSON object text from raw model responses.

- `vlm_descriptor.py`
  - Vision stage implementation:
    - Loads HF VLM + processor.
    - Runs inference on local image.
    - Parses response JSON.
    - Validates output against `TherapyFrameDescription`.
    - Includes retry strategy for transient generation failures.

- `llm_prompt_generator.py`
  - Text stage implementation:
    - Loads local vLLM model.
    - Builds prompt using structured frame JSON.
    - Generates Veo3 prompt bundle JSON.
    - Validates output against `Veo3PromptBundle`.

- `pipeline.py`
  - End-to-end orchestration:
    - Calls VLM descriptor then LLM prompt generator.
    - Handles output serialization to disk.

- `cli.py`
  - Typer-based CLI command surface for running the pipeline from terminal.

### Tests

- `tests/test_json_utils.py`
  - Unit test for robust JSON extraction from model output strings.

- `tests/test_schemas.py`
  - Unit test ensuring representative payloads validate against schema.

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
