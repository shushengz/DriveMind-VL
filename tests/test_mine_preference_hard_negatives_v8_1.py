from src.data.mine_preference_hard_negatives_v8_1 import mine


def row(sample_id, setting, prediction, gold="Yes."):
    return {"id": sample_id, "setting": setting, "prediction": prediction, "gold": gold, "question": "Is a car on the left?"}


def test_mine_uses_answer_only_control_failures():
    group = {
        "normal": row("a", "normal", '{"answer":"No."}'),
        "text_only": row("a", "text_only", '{"answer":"Yes."}'),
        "wrong_image": row("a", "wrong_image", '{"answer":"Yes."}'),
        "blank_image": row("a", "blank_image", '{"answer":"Yes."}'),
    }
    rows, stats = mine([group], ["a"], set())
    types = {item["hard_type"] for item in rows}
    assert {"text_prior_bias", "wrong_image_confound", "blank_prior_answer", "control_over_gold_overlap", "normal_wrong_anchor", "spatial_relation_error"} <= types
    assert stats["uses_model_predictions"] is True
