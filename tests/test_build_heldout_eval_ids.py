import json
from pathlib import Path

from src.eval.build_heldout_eval_ids import available_heldout_ids, build_payload


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_selects_only_aligned_unseen_ids(tmp_path):
    train = tmp_path / "train.jsonl"
    data = tmp_path / "visual"
    write_jsonl(train, [{"id": "a_normal_replay"}])
    for setting in ("normal", "text_only", "wrong_image", "blank_image"):
        write_jsonl(data / f"lingoqa_strict_{setting}.jsonl", [{"id": "a"}, {"id": "b"}])
    ids, counts = available_heldout_ids(train, data, "lingoqa")
    assert ids == ["b"]
    assert counts["heldout_available"] == 1
    payload = build_payload(ids, 100, 42, counts)
    assert payload["ids"] == ["b"]
    assert payload["warnings"]


def test_zero_candidates_warns():
    payload = build_payload([], 100, 42, {"heldout_available": 0})
    assert payload["selected"] == 0
    assert "no held-out ids" in payload["warnings"][-1]
