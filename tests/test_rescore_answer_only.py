
import json
from pathlib import Path
from src.eval.rescore_answer_only import build_rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


def test_rescore_answer_only_aligned(tmp_path):
    root = tmp_path / "predictions"
    for setting in ["normal", "text_only", "wrong_image", "blank_image"]:
        rows = [
            {"id":"a", "dataset":"lingoqa", "model_name":"m", "mode":"strict_visual", "setting":setting, "gold":"yes", "prediction":"{\"answer\":\"yes\",\"reason\":\"noise words\"}"},
            {"id":"b", "dataset":"lingoqa", "model_name":"m", "mode":"strict_visual", "setting":setting, "gold":"no", "prediction":"{\"answer\":\"no\"}"},
        ]
        write_jsonl(root / "lingoqa" / "m" / "strict_visual" / f"{setting}.jsonl", rows)
    rows, warnings = build_rows(root, "lingoqa", "strict_visual", ["m"], with_ci=False)
    assert not warnings
    answer = [r for r in rows if r["score_mode"] == "answer_only"][0]
    raw = [r for r in rows if r["score_mode"] == "raw_full"][0]
    assert answer["normal_f1"] == 1.0
    assert answer["parse_success_rate"] == 1.0
    assert raw["normal_f1"] < 1.0


def test_rescore_missing_setting_errors(tmp_path):
    root = tmp_path / "predictions"
    write_jsonl(root / "lingoqa" / "m" / "strict_visual" / "normal.jsonl", [{"id":"a","setting":"normal","gold":"yes","prediction":"yes"}])
    try:
        build_rows(root, "lingoqa", "strict_visual", ["m"], with_ci=False)
    except FileNotFoundError as exc:
        assert "text_only" in str(exc)
    else:
        raise AssertionError("expected clear missing setting error")
