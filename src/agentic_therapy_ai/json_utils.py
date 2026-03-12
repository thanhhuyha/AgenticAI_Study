from __future__ import annotations

import json
from typing import Any


def extract_json(raw_text: str) -> dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(raw_text[start : end + 1])
