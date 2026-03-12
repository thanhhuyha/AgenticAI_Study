from __future__ import annotations

import json
from vllm import LLM, SamplingParams

from .config import RuntimeSettings
from .json_utils import extract_json
from .prompts import VEO3_PROMPT_SYSTEM
from .schemas import TherapyFrameDescription, Veo3PromptBundle


class Veo3PromptGenerator:
    """Creates Veo3 prompt bundles from structured frame descriptions."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.model = LLM(
            model=settings.llm_model_id,
            tensor_parallel_size=settings.tensor_parallel_size,
            dtype=settings.dtype,
            trust_remote_code=True,
        )

    def build_prompt_bundle(self, description: TherapyFrameDescription) -> Veo3PromptBundle:
        input_payload = description.model_dump(mode="json")
        user_prompt = (
            "Generate a Veo3 prompt bundle from the following therapy frame JSON. "
            "Output JSON only with keys: base_prompt, positive_constraints, negative_constraints, "
            "variation_axes, output_format_notes.\n"
            f"INPUT_JSON:\n{json.dumps(input_payload, indent=2)}"
        )
        final_prompt = f"<|system|>\n{VEO3_PROMPT_SYSTEM}\n<|user|>\n{user_prompt}\n<|assistant|>"

        outputs = self.model.generate(
            [final_prompt],
            sampling_params=SamplingParams(
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                max_tokens=self.settings.max_new_tokens_llm,
            ),
        )
        raw = outputs[0].outputs[0].text
        parsed = extract_json(raw)
        return Veo3PromptBundle.model_validate(parsed)

