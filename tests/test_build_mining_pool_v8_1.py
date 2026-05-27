import json

from src.data.build_mining_pool_v8_1 import build_pool


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_pool_excludes_heldout_by_original_identity(tmp_path):
    vc = tmp_path / "vc"
    vc.mkdir()
    rows = [
        {"id": "train_a", "metadata": {"original_id": "shared"}},
        {"id": "train_b", "metadata": {"original_id": "safe"}},
    ]
    for setting in ("normal", "text_only", "wrong_image", "blank_image"):
        write_jsonl(vc / f"lingoqa_strict_{setting}.jsonl", rows)
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"ids": ["lingoqa_test_shared"]}), encoding="utf-8")
    payload, summary = build_pool(vc, heldout, max_ids=10, seed=1)
    assert payload["ids"] == ["train_b"]
    assert summary["heldout_removed"] == 1
    assert summary["leakage_after_filter"] == 0
