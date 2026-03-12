from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class LightingDirection(str, Enum):
    FRONT = "front"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    TOP = "top"
    BACK = "back"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LightingCondition(str, Enum):
    NATURAL = "natural"
    ARTIFICIAL = "artificial"
    MIXED = "mixed"
    LOW_LIGHT = "low_light"
    HARSH = "harsh"
    SOFT = "soft"
    UNKNOWN = "unknown"


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    HIGH_ANGLE = "high_angle"
    LOW_ANGLE = "low_angle"
    DUTCH = "dutch"
    UNKNOWN = "unknown"


class PoseLandmark(BaseModel):
    name: str = Field(description="Body landmark or keypoint label")
    position: str = Field(description="Natural-language position of the landmark")


class BodyPoseDescription(BaseModel):
    exercise_name: str = Field(description="Likely therapy exercise name")
    body_orientation: str = Field(description="Facing direction and torso orientation")
    weight_distribution: str = Field(description="How body weight is distributed")
    joint_angles_summary: str = Field(description="Approximate shoulder/elbow/hip/knee/ankle angles")
    balance_state: str = Field(description="Stable, unstable, assisted, etc.")
    landmark_notes: List[PoseLandmark] = Field(default_factory=list)
    potential_form_issues: List[str] = Field(default_factory=list)


class SubjectAppearance(BaseModel):
    clothing_upper: str
    clothing_lower: str
    footwear: str
    color_palette: List[str] = Field(default_factory=list)
    body_shape_estimate: str = Field(description="Neutral anthropometric description")


class ImageCondition(BaseModel):
    lighting_direction: LightingDirection
    lighting_condition: LightingCondition
    contrast_level: str = Field(description="low/medium/high + notes")
    sharpness: str = Field(description="blur level and details")
    camera_angle: CameraAngle
    camera_distance: str = Field(description="close-up, medium, wide")
    framing: str = Field(description="full body, half body, cropped")


class SceneContext(BaseModel):
    environment_type: str = Field(description="indoor/outdoor + room type")
    background_clutter_level: str = Field(description="low/medium/high")
    background_objects: List[str] = Field(default_factory=list)
    occlusions: List[str] = Field(default_factory=list)


class TherapyFrameDescription(BaseModel):
    schema_version: str = Field(default="1.0")
    subject_pose: BodyPoseDescription
    subject_appearance: SubjectAppearance
    image_condition: ImageCondition
    scene_context: SceneContext
    confidence: float = Field(ge=0.0, le=1.0)


class Veo3PromptBundle(BaseModel):
    base_prompt: str
    positive_constraints: List[str]
    negative_constraints: List[str]
    variation_axes: List[str]
    output_format_notes: str
