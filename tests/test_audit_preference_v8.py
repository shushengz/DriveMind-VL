from src.data.audit_preference_v8 import audit


def pair(pair_type, setting, source_id, chosen="Yes.", rejected="No."):
    return {
        "id": f"{source_id}_{pair_type}",
        "source_id": source_id,
        "pair_type": pair_type,
        "setting": setting,
        "prompt": "Q",
        "chosen": {"answer": chosen},
        "rejected": {"answer": rejected},
        "metadata": {"source_original_id": source_id + "_orig"},
    }


def test_preference_audit_accepts_clean_balanced_rows():
    rows = []
    rows += [pair("normal_anchor_gold_vs_bad", "normal", f"n{i}") for i in range(42)]
    rows += [pair("normal_gold_vs_refusal", "normal", f"r{i}", "Yes.", "Insufficient evidence to answer reliably.") for i in range(13)]
    rows += [pair("control_abstain_vs_hallucination", "blank_image", f"c{i}", "Insufficient evidence to answer reliably.", "Yes.") for i in range(22)]
    rows += [pair("wrong_image_caution_vs_confident_answer", "wrong_image", f"w{i}", "The provided input does not support a reliable answer.", "Yes.") for i in range(15)]
    rows += [pair("spatial_gold_vs_spatial_wrong", "normal", f"s{i}") for i in range(8)]
    report = audit(rows, set())
    assert report["train_ready"] is True
    assert report["normal_pair_ratio"] == 0.55
    assert report["control_pair_ratio"] == 0.37
    assert report["reason_field_count"] == 0
