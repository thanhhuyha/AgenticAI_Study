from agentic_therapy_ai.schemas import TherapyFrameDescription


def test_therapy_frame_description_validation() -> None:
    payload = {
        "schema_version": "1.0",
        "subject_pose": {
            "exercise_name": "single leg balance",
            "body_orientation": "facing camera",
            "weight_distribution": "mostly on right leg",
            "joint_angles_summary": "right knee 10deg flexed, left knee 70deg flexed",
            "balance_state": "stable",
            "landmark_notes": [{"name": "left_ankle", "position": "raised behind right ankle"}],
            "potential_form_issues": ["slight trunk lean"],
        },
        "subject_appearance": {
            "clothing_upper": "black t-shirt",
            "clothing_lower": "gray joggers",
            "footwear": "barefoot",
            "color_palette": ["black", "gray", "beige"],
            "body_shape_estimate": "average build adult",
        },
        "image_condition": {
            "lighting_direction": "side_left",
            "lighting_condition": "natural",
            "contrast_level": "medium",
            "sharpness": "high",
            "camera_angle": "eye_level",
            "camera_distance": "medium",
            "framing": "full body",
        },
        "scene_context": {
            "environment_type": "indoor living room",
            "background_clutter_level": "medium",
            "background_objects": ["sofa", "coffee table"],
            "occlusions": [],
        },
        "confidence": 0.89,
    }
    model = TherapyFrameDescription.model_validate(payload)
    assert model.subject_pose.exercise_name == "single leg balance"
