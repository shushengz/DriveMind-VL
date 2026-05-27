import json

from src.rl.reward_driving_vqa import compute_reward, parse_answer_json, reward_hacking_warnings


def test_parse_answer_json_and_format_reward():
    obj, err = parse_answer_json(json.dumps({"answer": "green", "reason": "seen"}))
    assert not err
    assert obj["answer"] == "green"
    reward = compute_reward({"id": "1", "setting": "normal", "prediction": json.dumps(obj), "gold": "green"})
    assert reward["r_format"] == 1.0
    assert reward["r_answer"] == 1.0


def test_control_hallucination_penalty_and_refusal_reward():
    bad = compute_reward({"id": "1", "setting": "blank_image", "prediction": "the image shows a car", "gold": "no"})
    good = compute_reward({"id": "1", "setting": "blank_image", "prediction": bytes.fromhex("e7bcbae5b091e59bbee5838fe4bfa1e681afefbc8ce697a0e6b395e58fafe99da0e588a4e696ade38082").decode("utf-8"), "gold": "no"})
    assert bad["hallucination"]
    assert bad["r_control_calibration"] < good["r_control_calibration"]


def test_reward_hacking_warning():
    rewards = [{"setting": "normal", "f1": 0.1, "refusal": True, "answer_length": 10, "reward": -1.0} for _ in range(4)]
    warnings = reward_hacking_warnings(rewards)
    assert any("normal refusal" in w for w in warnings)
