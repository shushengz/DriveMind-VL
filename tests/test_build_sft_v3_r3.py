
import json
from pathlib import Path
from src.data.build_sft_v3_r3 import read_groups, build_candidates, sample_rows, stats, FORBIDDEN_CONTROL_RE


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


def make_row(i, setting):
    return {"id":f"id{i}","dataset":"lingoqa","mode":"strict_visual","setting":setting,"question":"Is there a pedestrian on the left?","gold":"yes","image_paths":["x.jpg"] if setting != "text_only" else [],"image_labels":["[Frame 0]"],"prompt":"Q","metadata":{}}


def test_build_r3_ratios_and_answer_only(tmp_path):
    vc=tmp_path / "vc"
    for setting in ["normal","blank_image","wrong_image","text_only"]:
        write_jsonl(vc / f"lingoqa_strict_{setting}.jsonl", [make_row(i, setting) for i in range(60)])
    groups=read_groups(vc)
    rows,warnings=sample_rows(build_candidates(groups), seed=1, max_samples=80)
    st=stats(rows,warnings)
    assert st["train_ready"]
    assert st["normal_like_ratio"] >= 0.75
    assert st["control_ratio"] <= 0.15
    assert st["reason_field_count"] == 0
    for row in rows:
        assert set(row["assistant"].keys()) == {"answer"}
        if row["sample_type"].endswith("calibration"):
            assert not FORBIDDEN_CONTROL_RE.search(row["assistant"]["answer"])
