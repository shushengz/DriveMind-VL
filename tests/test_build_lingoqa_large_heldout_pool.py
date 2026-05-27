from src.eval.build_lingoqa_large_heldout_pool import normalize_id, select_payload


def test_normalize_strips_source_and_generated_suffixes():
    assert normalize_id("lingoqa_eval_000001_normal_replay") == "lingoqa_eval_000001"
    assert normalize_id("lingoqa_eval_000001_blank_image") == "lingoqa_eval_000001"
    assert normalize_id("generated", {"source_id": "lingoqa_test_123_wrong_image"}) == "lingoqa_test_123"


def test_select_payload_does_not_invent_large_pool():
    payload = select_payload(["a", "b"], 300, 42, pinned=["a"])
    assert payload["ids"][0] == "a"
    assert payload["selected"] == 2
    assert not payload["is_full_size"]
    assert payload["warnings"]
