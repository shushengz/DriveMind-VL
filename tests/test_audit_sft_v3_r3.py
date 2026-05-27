
from src.data.audit_sft_v3_r3 import audit


def row(sample_type, answer):
    return {"id":sample_type+answer, "sample_type":sample_type, "assistant":{"answer":answer}, "messages":[{"role":"assistant","content":"{}"}]}


def test_audit_passes_good_data():
    rows=[]
    rows += [row("normal_replay", "yes") for _ in range(70)]
    rows += [row("spatial_normal_qa", "left") for _ in range(10)]
    rows += [row("blank_calibration", "Insufficient evidence to answer reliably.") for _ in range(5)]
    rows += [row("wrong_image_calibration", "The provided input does not support a reliable answer.") for _ in range(5)]
    rows += [row("text_only_calibration", "Insufficient evidence to answer reliably.") for _ in range(4)]
    report=audit(rows)
    assert report["train_ready"]
    assert report["reason_field_count"] == 0


def test_audit_fails_reason_and_forbidden_terms():
    rows=[row("normal_replay", "yes") for _ in range(80)] + [row("blank_calibration", "The image is blank") for _ in range(10)]
    rows[0]["assistant"]["reason"] = "bad"
    report=audit(rows)
    assert not report["train_ready"]
    assert report["reason_field_count"] == 1
    assert report["control_forbidden_visual_terms_rate"] > 0
