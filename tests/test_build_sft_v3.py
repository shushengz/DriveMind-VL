from src.data.build_sft_v3 import build_for_dataset, stats


def source_row(idx="x1", paths=None):
    return {
        "id": idx,
        "instruction": "Is the vehicle on the left?",
        "answer": {"answer": "Yes."},
        "meta": {"benchmark_source": "lingoqa", "external": {"image_paths": paths or ["a.jpg", "b.jpg"]}},
    }


def test_build_sft_v3_rows_and_stats():
    rows = build_for_dataset([source_row(), source_row("x2", ["c.jpg", "d.jpg"])], "lingoqa", 1)
    assert any(r["sample_type"] == "normal_visual_qa" for r in rows)
    assert any(r["sample_type"] == "blank_refusal" for r in rows)
    assert any(r["sample_type"] == "spatial_hard_negative" for r in rows)
    for row in rows:
        assert set(row["assistant"].keys()) == {"answer", "reason"}
        assert "references" not in row["assistant"]
    st = stats(rows)
    assert st["total"] == len(rows)
    assert st["refusal_sample_ratio"] > 0
