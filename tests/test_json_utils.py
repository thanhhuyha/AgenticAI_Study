from agentic_therapy_ai.json_utils import extract_json


def test_extract_json_with_prefix_and_suffix() -> None:
    raw = "header {\"k\": 1, \"v\": [1,2]} trailer"
    result = extract_json(raw)
    assert result == {"k": 1, "v": [1, 2]}
