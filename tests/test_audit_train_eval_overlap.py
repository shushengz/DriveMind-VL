import json
from pathlib import Path

from src.eval.audit_train_eval_overlap import audit_overlap, normalize_id


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_normalize_id_removes_r3_suffix_or_uses_source_id():
    assert normalize_id("lingoqa_eval_000171_normal_replay") == "lingoqa_eval_000171"
    assert normalize_id("ignored", {"metadata": {"source_id": "lingoqa_eval_000003"}}) == "lingoqa_eval_000003"


def test_audit_reports_setting_overlap(tmp_path):
    train = tmp_path / "train.jsonl"
    prediction_dir = tmp_path / "predictions"
    write_jsonl(train, [{"id": "a_normal_replay"}, {"id": "b_blank_calibration"}])
    for setting in ("normal", "text_only", "wrong_image", "blank_image"):
        write_jsonl(prediction_dir / f"{setting}.jsonl", [{"id": "a"}, {"id": "c"}])
    result = audit_overlap(train, prediction_dir)
    assert result["train_unique_ids"] == 2
    assert result["eval_unique_ids"] == 2
    assert result["overlap_ids"] == ["a"]
    assert result["overlap_rate_eval"] == 0.5
    assert result["by_setting"]["normal"]["overlap_count"] == 1
