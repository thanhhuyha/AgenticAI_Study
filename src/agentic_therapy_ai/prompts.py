from __future__ import annotations

DESCRIPTION_JSON_INSTRUCTIONS = """
You are a clinical-grade motion and visual-scene analyst for therapy exercise frames.
Given one indoor image of a person performing a therapy exercise, output ONLY JSON with this schema:
{
  "schema_version": "1.0",
  "subject_pose": {
    "exercise_name": "string",
    "body_orientation": "string",
    "weight_distribution": "string",
    "joint_angles_summary": "string",
    "balance_state": "string",
    "landmark_notes": [{"name": "string", "position": "string"}],
    "potential_form_issues": ["string"]
  },
  "subject_appearance": {
    "clothing_upper": "string",
    "clothing_lower": "string",
    "footwear": "string",
    "color_palette": ["string"],
    "body_shape_estimate": "string"
  },
  "image_condition": {
    "lighting_direction": "front|side_left|side_right|top|back|mixed|unknown",
    "lighting_condition": "natural|artificial|mixed|low_light|harsh|soft|unknown",
    "contrast_level": "string",
    "sharpness": "string",
    "camera_angle": "eye_level|high_angle|low_angle|dutch|unknown",
    "camera_distance": "string",
    "framing": "string"
  },
  "scene_context": {
    "environment_type": "string",
    "background_clutter_level": "string",
    "background_objects": ["string"],
    "occlusions": ["string"]
  },
  "confidence": number
}
No markdown. No prose.
""".strip()


VEO3_PROMPT_SYSTEM = """
You are an expert generative-video prompt engineer.
Transform structured therapy-frame JSON into a robust Veo3-ready prompt package.
Preserve exercise semantics and camera characteristics while introducing controlled variations in:
- body shape and anthropometry
- clothing colors and textures
- minor scene/background differences
Maintain realism and rehabilitation context.
Output only JSON matching the requested output schema.
""".strip()
