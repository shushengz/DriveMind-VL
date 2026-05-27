import pytest

from src.eval.metrics_visual_control import exact_match, is_control_hallucination, is_refusal, summarize_rows, token_f1


def row(sample_id, setting, pred, gold="green light", f1=None):
    out = {"id": sample_id, "dataset": "lingoqa", "model_name": "m", "mode": "strict_visual", "setting": setting, "prediction": pred, "gold": gold}
    if f1 is not None:
        out["f1"] = f1
    return out


def test_f1_and_em():
    assert token_f1("green light", "green light") == 1.0
    assert token_f1("green", "green light") > 0
    assert exact_match("Green light!", "green light") == 1.0


def test_case_and_setting_gap():
    rows = [row("1", "normal", "green light", f1=1.0), row("1", "text_only", "red", f1=0.0), row("1", "wrong_image", "red", f1=0.0), row("1", "blank_image", "red", f1=0.0)]
    summary, cases = summarize_rows(rows)
    assert summary["setting_gap"] == 1.0
    assert summary["case_gap"] == 1.0
    assert cases[0]["failure_type"] == "visual_gain"


def test_refusal_detection():
    assert is_refusal(bytes.fromhex("e7bcbae5b091e59bbee5838fe4bfa1e681afefbc8ce697a0e6b395e58fafe99da0e588a4e696ade38082").decode("utf-8"))
    assert is_refusal("cannot determine from the image")


def test_hallucination_detection():
    assert is_control_hallucination("the image shows a car on the left", "blank_image")
    assert not is_control_hallucination("not enough visual information to tell", "blank_image")


def test_missing_setting_error():
    rows = [row("1", "normal", "green", f1=1.0), row("1", "text_only", "green", f1=1.0)]
    with pytest.raises(ValueError, match="missing visual-control settings"):
        summarize_rows(rows)
