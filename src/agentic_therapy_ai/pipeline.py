from __future__ import annotations

from pathlib import Path

import orjson

from .config import RuntimeSettings
from .llm_prompt_generator import Veo3PromptGenerator
from .schemas import TherapyFrameDescription, Veo3PromptBundle
from .vlm_descriptor import VLMDescriptor


class TherapyAgentPipeline:
    """End-to-end orchestrator for frame description and synthetic prompt generation."""

    def __init__(
        self,
        settings: RuntimeSettings,
        descriptor: VLMDescriptor | None = None,
        prompt_generator: Veo3PromptGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.descriptor = descriptor or VLMDescriptor(settings)
        self.prompt_generator = prompt_generator or Veo3PromptGenerator(settings)

    def run(self, image_path: Path, run_id: str) -> tuple[TherapyFrameDescription, Veo3PromptBundle]:
        description = self.descriptor.describe_frame(image_path)
        prompt_bundle = self.prompt_generator.build_prompt_bundle(description)
        self._write_outputs(run_id, description, prompt_bundle)
        return description, prompt_bundle

    def _write_outputs(
        self,
        run_id: str,
        description: TherapyFrameDescription,
        prompt_bundle: Veo3PromptBundle,
    ) -> None:
        output_dir = self.settings.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_description.json").write_bytes(
            orjson.dumps(description.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )
        (output_dir / "veo3_prompt_bundle.json").write_bytes(
            orjson.dumps(prompt_bundle.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )
