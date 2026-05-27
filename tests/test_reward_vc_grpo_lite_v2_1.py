from src.rl.reward_vc_grpo_lite_v2_1 import compute_pairwise_preference_reward, compute_reward_v2_1


def row(setting, prediction, gold="stop", dataset="lingoqa", question="", labels=None):
    return {"id": "x", "setting": setting, "prediction": prediction, "gold": gold, "dataset": dataset, "question": question, "image_labels": labels or []}


def rec(normal, text, wrong, blank, gold="stop", dataset="lingoqa", question="", capability="", tags=None, labels=None):
    return {"id": "x", "dataset": dataset, "model_name": "test", "question": question, "capability": capability, "failure_tags": tags or [], "image_labels": labels or [], "settings": {
        "normal": row("normal", normal, gold, dataset, question, labels),
        "text_only": row("text_only", text, gold, dataset, question),
        "wrong_image": row("wrong_image", wrong, gold, dataset, question, labels),
        "blank_image": row("blank_image", blank, gold, dataset, question, labels),
    }}


def test_blank_penalty_is_tiered_and_saturated():
    medium = compute_reward_v2_1(rec("stop", "insufficient evidence", "insufficient evidence", "stop now", "stop"))
    high = compute_reward_v2_1(rec("stop", "insufficient evidence", "insufficient evidence", "stop", "stop"))
    assert medium["blank_penalty_tier"] in {"medium", "high"}
    assert high["blank_high_f1_penalty"] <= 0.65 * 0.90 + 1e-9


def test_camera_label_alone_does_not_penalize_low_wrong_image():
    result = compute_reward_v2_1(rec("stop", "insufficient evidence", "different", "insufficient evidence", "stop", "drivelm", "What is visible?", labels=["[CAM_FRONT]"]))
    assert result["camera_penalty"] == 0


def test_camera_specific_wrong_high_penalized():
    result = compute_reward_v2_1(rec("stop", "insufficient evidence", "stop", "insufficient evidence", "stop", "drivelm", "What is visible in CAM_FRONT?", tags=["camera_specific_failure"], labels=["[CAM_FRONT]"]))
    assert result["camera_penalty"] > 0


def test_object_control_high_penalized():
    result = compute_reward_v2_1(rec("car", "car", "car", "insufficient evidence", "car", "drivelm", "What is <c2,CAM_FRONT,1,2>?", "object_token", ["object_token_failure"], ["[CAM_FRONT]"]))
    assert result["object_penalty"] > 0


def test_object_token_low_control_not_over_penalized():
    result = compute_reward_v2_1(rec("car", "insufficient evidence", "insufficient evidence", "insufficient evidence", "car", "drivelm", "What is <c2,CAM_FRONT,1,2>?", "object_token", ["object_token_failure"], ["[CAM_FRONT]"]))
    assert result["object_penalty"] == 0


def test_correct_with_control_caution_is_high():
    assert compute_reward_v2_1(rec("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))["total_reward"] > 1.0


def test_normal_refusal_remains_low():
    assert compute_reward_v2_1(rec("insufficient evidence", "insufficient evidence", "insufficient evidence", "insufficient evidence"))["total_reward"] < 0


def test_pairwise_prefers_good_case():
    good = rec("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence")
    bad = rec("stop", "stop", "stop", "stop")
    assert compute_pairwise_preference_reward(good, bad)["preferred"] == "a"
