from src.data.audit_preference_v8_2 import audit, output_valid
from src.data.build_preference_v8_2 import quotas


def test_v8_2_quotas_preserve_requested_balance():
    target = quotas(500)
    assert target["normal_anchor_gold_vs_model_wrong"] == 175
    assert target["normal_gold_vs_refusal"] == 25
    assert target["blank_high_f1_vs_caution"] == 75
    assert sum(target.values()) == 500


def test_answer_only_output_validator_rejects_reason():
    assert output_valid({"answer": "No."})
    assert not output_valid({"answer": "No.", "reason": "extra"})


def test_audit_blocks_heldout_pair():
    pair = {
        "id": "held_control",
        "source_id": "held",
        "pair_type": "control_direct_answer_vs_caution",
        "setting": "blank_image",
        "chosen": {"answer": "Insufficient evidence to answer reliably."},
        "rejected": {"answer": "Yes."},
        "metadata": {"rejected_from_actual_prediction": True},
    }
    candidate = {"candidate_type": "control_direct_answer"}
    report = audit([pair], [candidate], {"held"})
    assert report["heldout_leakage_count"] == 1
    assert report["train_ready"] is False
