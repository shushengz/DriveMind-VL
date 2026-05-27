from src.data.drivemind_vl_schema import DriveMindRecord, answer_only_json, extract_answer, from_sft_row, is_answer_only_json, leakage_ids, make_ms_swift_user_text, make_user_content, normalize_id


def test_normalize_id_strips_training_and_setting_suffixes():
    assert normalize_id("lingoqa_eval_0001_blank_calibration") == "lingoqa_eval_0001"
    assert normalize_id("x_wrong_image") == "x"


def test_sft_conversion_extracts_answer_only_and_control_setting():
    row = {"id": "x_text_only_calibration", "dataset": "lingoqa", "sample_type": "text_only_calibration", "prompt": "q", "messages": [{"role": "assistant", "content": '{"answer":"insufficient evidence"}'}], "image_paths": []}
    record = from_sft_row(row)
    assert record.setting == "text_only"
    assert record.answer == "insufficient evidence"
    assert record.metadata["is_control"] is True


def test_multi_image_and_text_only_content():
    normal = DriveMindRecord("a", "lingoqa", "normal", "q", "a", ["1.jpg", "2.jpg"], ["one", "two"], {})
    control = DriveMindRecord("b", "lingoqa", "text_only", "q", "a", ["ignored.jpg"], [], {})
    assert len(make_user_content(normal)) == 3
    assert make_user_content(control) == [{"type": "text", "text": "q"}]
    assert make_ms_swift_user_text(normal) == "<image><image>q"
    assert make_ms_swift_user_text(control) == "q"


def test_leakage_uses_normalized_source_identity():
    record = DriveMindRecord("x_blank_calibration", "lingoqa", "blank_image", "q", "a", [], [], {"source_id": "x"})
    assert leakage_ids([record], {"x"}) == ["x"]


def test_extract_answer_accepts_object_or_json_string():
    assert extract_answer({"answer": "yes"}) == "yes"
    assert extract_answer('{"answer":"no"}') == "no"
    assert is_answer_only_json(answer_only_json("stop"))
