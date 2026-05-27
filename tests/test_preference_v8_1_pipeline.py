from collections import Counter

from src.data.audit_preference_v8_1 import audit
from src.data.build_preference_v8_1 import build_pairs


def hard(source_id, hard_type, setting):
    return {
        "id": f"{source_id}_{hard_type}",
        "source_id": source_id,
        "hard_type": hard_type,
        "setting": setting,
        "gold": "Gold answer.",
        "normal_answer": "Wrong normal.",
        "text_only_answer": "Text guess.",
        "wrong_image_answer": "Wrong image guess.",
        "blank_image_answer": "Blank guess.",
        "trigger_reason": "test",
    }


def test_model_mined_pairs_pass_clean_audit():
    specification = [
        ("text_prior_bias", "text_only", 22),
        ("blank_prior_answer", "blank_image", 18),
        ("wrong_image_confound", "wrong_image", 17),
        ("control_over_gold_overlap", "text_only", 10),
        ("normal_wrong_anchor", "normal", 26),
        ("spatial_relation_error", "normal", 7),
    ]
    hard_rows = []
    visual = {setting: {} for setting in ("normal", "text_only", "wrong_image", "blank_image")}
    index = 0
    for hard_type, hard_setting, count in specification:
        for _ in range(count):
            sample_id = f"sample_{index:03d}"
            index += 1
            hard_rows.append(hard(sample_id, hard_type, hard_setting))
            for setting in visual:
                visual[setting][sample_id] = {
                    "id": sample_id,
                    "prompt": "Prompt",
                    "image_paths": [],
                    "image_labels": [],
                }
    pairs, _ = build_pairs(hard_rows, visual, set(), max_pairs=100, seed=1)
    lookup = {(row["source_id"], row["hard_type"]): row for row in hard_rows}
    report = audit(pairs, lookup, Counter(row["hard_type"] for row in hard_rows), set())
    assert report["train_ready"] is True
    assert report["rejected_from_r3_actual_prediction_rate"] == 1.0
    assert report["heldout_leakage_count"] == 0
