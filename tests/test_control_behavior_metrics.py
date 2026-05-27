from src.eval.control_behavior_metrics import (
    gold_overlap_high,
    is_action_answer,
    is_count_answer,
    is_direct_answer,
    is_refusal_or_caution,
    is_short_prior_answer,
)


def test_caution_detection_english_and_chinese():
    assert is_refusal_or_caution("Insufficient evidence to answer reliably.")
    assert is_refusal_or_caution("信息不足，无法可靠判断。")
    assert not is_refusal_or_caution("Yes")


def test_direct_and_short_prior_answers():
    assert is_direct_answer("Yes")
    assert is_direct_answer("Slow down.")
    assert is_short_prior_answer("Turn left")
    assert is_short_prior_answer("2")
    assert not is_direct_answer("The provided input does not support a reliable answer.")


def test_action_and_count_answers():
    assert is_action_answer("Maintain speed.")
    assert is_action_answer("Change lane")
    assert is_count_answer("No pedestrians")
    assert is_count_answer("3")
    assert not is_count_answer("There are three cars ahead.")


def test_high_overlap_thresholds():
    assert gold_overlap_high(0.20)
    assert gold_overlap_high(0.30, threshold=0.30)
    assert not gold_overlap_high(0.19)
