from src.data.build_preference_v8 import build_pairs, stats


def visual_row(sample_id, setting="normal", question="Is the vehicle on the left lane?"):
    return {
        "id": sample_id,
        "dataset": "lingoqa",
        "setting": setting,
        "prompt": "Q",
        "question": question,
        "gold": "Yes.",
        "image_paths": [],
        "image_labels": [],
        "metadata": {"original_id": f"orig_{sample_id}", "setting": setting},
    }


def test_preference_v8_pairs_and_stats():
    ids = [f"x{i}" for i in range(30)]
    rows = {setting: [visual_row(sample_id, setting) for sample_id in ids] for setting in ("normal", "text_only", "wrong_image", "blank_image")}
    pairs = build_pairs(rows, set(), max_pairs=20, seed=1)
    assert pairs
    assert all({"answer"} == set(p["chosen"].keys()) for p in pairs)
    assert all({"answer"} == set(p["rejected"].keys()) for p in pairs)
    assert any(p["pair_type"] == "control_abstain_vs_hallucination" for p in pairs)
    st = stats(pairs, [], uses_model_predictions=False)
    assert st["total_pairs"] == len(pairs)
    assert st["weight_sum"] > 0
    assert st["uses_model_predictions"] is False
