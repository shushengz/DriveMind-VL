from src.data.exclude_heldout_ids import filter_candidates, identity_keys, normalize_id


def test_excludes_matching_original_identity():
    heldout_keys = identity_keys("lingoqa_test_hash_1")
    rows = [
        {"id": "lingoqa_eval_001", "metadata": {"original_id": "hash_1"}},
        {"id": "lingoqa_eval_002", "metadata": {"original_id": "hash_2"}},
    ]
    kept, removed = filter_candidates(rows, heldout_keys)
    assert [row["id"] for row in kept] == ["lingoqa_eval_002"]
    assert [row["id"] for row in removed] == ["lingoqa_eval_001"]


def test_normalize_strips_pair_suffix():
    assert normalize_id("lingoqa_eval_001_normal_anchor_gold_vs_bad") == "lingoqa_eval_001"
