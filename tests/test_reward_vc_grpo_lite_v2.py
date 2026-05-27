from src.rl.reward_vc_grpo_lite_v2 import compute_reward_v2


def row(setting, prediction, gold="stop", question="", dataset="lingoqa", labels=None):
    return {
        "id": "c1", "setting": setting, "prediction": prediction, "gold": gold,
        "question": question, "dataset": dataset, "image_labels": labels or [],
    }


def record(normal, text, wrong, blank, gold="stop", dataset="lingoqa", question="", capability="", tags=None, labels=None):
    return {
        "id": "c1", "dataset": dataset, "model_name": "test", "question": question,
        "capability": capability, "failure_tags": tags or [], "image_labels": labels or [],
        "settings": {
            "normal": row("normal", normal, gold, question, dataset, labels),
            "text_only": row("text_only", text, gold, question, dataset),
            "wrong_image": row("wrong_image", wrong, gold, question, dataset, labels),
            "blank_image": row("blank_image", blank, gold, question, dataset, labels),
        },
    }


def test_normal_correct_control_caution_has_high_reward():
    good = compute_reward_v2(record("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    assert good["normal_reward"] > 1.0
    assert good["total_reward"] > 0.8


def test_blank_high_f1_lowers_reward():
    good = compute_reward_v2(record("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    bad = compute_reward_v2(record("stop", "insufficient evidence", "insufficient evidence", "stop"))
    assert bad["blank_high_f1_penalty"] > 0
    assert good["total_reward"] > bad["total_reward"]


def test_text_direct_answer_is_penalized():
    result = compute_reward_v2(record("stop", "stop", "insufficient evidence", "insufficient evidence"))
    assert result["text_direct_penalty"] > 0


def test_wrong_image_confound_is_penalized():
    result = compute_reward_v2(record("stop", "insufficient evidence", "stop", "insufficient evidence"))
    assert result["wrong_image_penalty"] > 0
    assert "wrong_image_confound" in result["reward_tags"]


def test_normal_refusal_is_penalized():
    result = compute_reward_v2(record("insufficient evidence", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    assert result["normal_refusal_penalty"] > 0
    assert result["total_reward"] < 0


def test_spatial_failure_gets_spatial_penalty():
    result = compute_reward_v2(record("left", "left", "left", "left", "right", "drivelm", "Is the car on the left lane?", "spatial_relation", ["spatial_relation_failure"], ["[CAM_FRONT]"]))
    assert result["spatial_penalty"] > 0


def test_object_token_failure_gets_object_penalty():
    result = compute_reward_v2(record("car", "car", "car", "car", "car", "drivelm", "What is <c2,CAM_FRONT,1,2>?", "object_token", ["object_token_failure"], ["[CAM_FRONT]"]))
    assert result["object_penalty"] > 0


def test_camera_specific_wrong_image_gets_camera_penalty():
    result = compute_reward_v2(record("stop", "insufficient evidence", "stop", "insufficient evidence", "stop", "drivelm", "What is seen in CAM_FRONT?", "camera_specific", ["camera_specific_failure"], ["[CAM_FRONT]"]))
    assert result["camera_penalty"] > 0
