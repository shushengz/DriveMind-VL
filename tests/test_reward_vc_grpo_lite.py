from src.rl.reward_vc_grpo_lite import compute_case_reward


def row(setting, pred, gold="stop"):
    return {"id": "c1", "setting": setting, "prediction": pred, "gold": gold}


def group(normal, text, wrong, blank, gold="stop"):
    return {
        "normal": row("normal", normal, gold),
        "text_only": row("text_only", text, gold),
        "wrong_image": row("wrong_image", wrong, gold),
        "blank_image": row("blank_image", blank, gold),
    }


def test_good_visual_dependency_beats_blank_prior_answer():
    good = compute_case_reward(group("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    bad = compute_case_reward(group("stop", "insufficient evidence", "insufficient evidence", "stop"))
    assert good["total_reward"] > bad["total_reward"]
    assert "blank_high_f1" in bad["reward_tags"]


def test_normal_refusal_is_strongly_penalized():
    correct = compute_case_reward(group("stop", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    refusal = compute_case_reward(group("insufficient evidence", "insufficient evidence", "insufficient evidence", "insufficient evidence"))
    assert correct["total_reward"] > refusal["total_reward"]
    assert refusal["normal_refusal_penalty"] > 0


def test_wrong_image_confound_is_penalized():
    clean = compute_case_reward(group("stop", "insufficient evidence", "different", "insufficient evidence"))
    confounded = compute_case_reward(group("stop", "insufficient evidence", "stop", "insufficient evidence"))
    assert clean["total_reward"] > confounded["total_reward"]
