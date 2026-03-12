from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import RuntimeSettings
from .json_utils import extract_json
from .prompts import DESCRIPTION_JSON_INSTRUCTIONS
from .schemas import TherapyFrameDescription


class VLMDescriptor:
    """Generates deterministic, schema-constrained frame descriptions from an image."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.processor = AutoProcessor.from_pretrained(settings.vlm_model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            settings.vlm_model_id,
            torch_dtype=getattr(torch, settings.dtype),
            device_map="auto",
            trust_remote_code=True,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def describe_frame(self, image_path: Path) -> TherapyFrameDescription:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=DESCRIPTION_JSON_INSTRUCTIONS,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens_vlm,
                do_sample=False,
            )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        parsed = extract_json(decoded)
        return TherapyFrameDescription.model_validate(parsed)

