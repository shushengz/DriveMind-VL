from src.rl.reward_vc_grpo_lite_v2_2 import compute_pairwise_preference_reward, compute_reward_v2_2


def row(setting, prediction, gold="stop", dataset="lingoqa", question=""):
    return {"id": "x", "setting": setting, "prediction": prediction, "gold": gold, "dataset": dataset, "question": question}


def rec(normal, text, wrong, blank, gold="stop", dataset="lingoqa", question="", capability="", tags=None):
    return {"id": "x", "dataset": dataset, "model_name": "test", "question": question, "capability": capability, "failure_tags": tags or [], "settings": {
        "normal": row("normal", normal, gold, dataset, question),
        "text_only": row("text_only", text, gold, dataset, question),
        "wrong_image": row("wrong_image", wrong, gold, dataset, question),
        "blank_image": row("blank_image", blank, gold, dataset, question),
    }}


def test_object_token_invariant_answer_is_penalized():
    result = compute_reward_v2_2(rec("slow down", "slow down", "slow down", "slow down", "stop", "drivelm", "What should <c1,CAM_FRONT> do?", "object_token", ["object_token_failure"]))
    assert result["object_token_invariant_triggered"]
    assert result["object_token_invariant_penalty"] > 0


def test_object_token_with_control_caution_is_not_invariant():
    result = compute_reward_v2_2(rec("slow down", "insufficient evidence", "insufficient evidence", "insufficient evidence", "slow down", "drivelm", "What should <c1,CAM_FRONT> do?", "object_token", ["object_token_failure"]))
    assert not result["object_token_invariant_triggered"]
    assert result["object_token_invariant_penalty"] == 0


def test_object_action_template_with_rewording_is_invariant():
    result = compute_reward_v2_2(rec("wait and stop", "slow down or stop", "brake and stop", "slow down", "turn", "drivelm", "What actions cause collision with <c1,CAM_FRONT>?", "object_token", ["object_token_failure"]))
    assert result["object_token_semantic_action_invariant_triggered"]
    assert result["object_token_invariant_penalty"] > 0


def test_cyclist_vs_car_is_category_mismatch():
    result = compute_reward_v2_2(rec("a car is present", "unknown", "unknown", "unknown", "a cyclist is present"))
    assert result["normal_object_category_mismatch_triggered"]
    assert result["normal_object_category_mismatch_penalty"] > 0


def test_vehicle_vs_car_is_not_severe_mismatch():
    result = compute_reward_v2_2(rec("a car is present", "unknown", "unknown", "unknown", "a vehicle is present"))
    assert not result["normal_object_category_mismatch_triggered"]


def test_terminate_is_invalid_generic_answer():
    result = compute_reward_v2_2(rec("stop", "terminate", "unknown", "unknown", "stop"))
    assert result["invalid_generic_answer_triggered"]
    assert result["invalid_generic_answer_penalty"] > 0


def test_control_same_as_normal_is_penalized():
    result = compute_reward_v2_2(rec("stop", "unknown", "unknown", "stop", "stop"))
    assert result["control_same_as_normal_triggered"]
    assert "blank_image" in result["same_as_normal_settings"]


def test_correct_normal_with_control_caution_remains_high():
    assert compute_reward_v2_2(rec("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))["total_reward"] > 1.0


def test_normal_refusal_remains_low():
    assert compute_reward_v2_2(rec("insufficient evidence", "insufficient evidence", "insufficient evidence", "insufficient evidence"))["total_reward"] < 0


def test_output_is_backward_compatible_and_pairwise_prefers_good():
    good = rec("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence")
    bad = rec("stop", "stop", "stop", "stop")
    result = compute_reward_v2_2(good)
    for key in ("normal_reward", "blank_penalty_tier", "camera_grounding_triggered", "object_grounding_triggered", "pairwise_ready"):
        assert key in result
    assert compute_pairwise_preference_reward(good, bad)["preferred"] == "a"
